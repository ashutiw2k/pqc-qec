# Multiprocessing Finetuning Hang Investigation — 2025-10-31

## Summary

- **Observed issue:** `scripts/finetune_transformer_predictions_mp.py` stalled after running ~16 hours; parent and worker processes remained alive but idle.
- **Impact:** Only 20 675 of 59 049 circuits were processed; remaining ~38 k circuits never started.
- **Root cause:** macOS terminated the pool’s feeder thread repeatedly with `EXC_RESOURCE` when the job exceeded the per-process Mach port limit (~267 k) while enqueuing tens of thousands of pending `imap` tasks. Once semaphore creation failed, no new work reached the workers, leaving them blocked.
- **Resolution plan:** Restart the job with bounded submission (batched input or larger `chunksize`), add watchdog monitoring, and instrument pool logging to detect feeder-thread exits promptly.

## Environment Snapshot

- **Host:** macOS 15.6.1 (24G90)
- **Python:** `/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python`
- **Command under investigation:**
  ```bash
  python scripts/finetune_transformer_predictions_mp.py \
    -i nogit_circuitdata/aug_angle_predictor_1022_start_subcircuits_gb10_20251028_093406_len10_enumall_100K_Nplus3.jsonl \
    -o nogit/finetune_results -q 1 -g 10 -k 10 -n 2000 -t 100 -b 10 -e 2 -p 10 --restart
  ```

## Timeline & Actions

1. **Process enumeration:** Confirmed parent and workers alive yet idle via `ps`.
2. **Stack sampling:** Captured `sample` traces for parent (`PID 35146`) and worker (`PID 39358`).
3. **Progress audit:** Compared output record count vs. input dataset size.
4. **System logs review:** Parsed recent `ExcResource_Python*.diag` reports.
5. **Conclusion:** Identified semaphore/port exhaustion leading to stalled pool.

## Evidence

### Process Snapshot

```bash
$ ps -p 35146,39358,39359,39360,39361,39362,39363,39364,39366,39367,39368 -o pid,stat,%cpu,%mem,etime,time,command
  PID STAT  %CPU %MEM  ELAPSED      TIME COMMAND
35146 S+     0.0  0.9 17:03:34   0:18.12 /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python scripts/finetune_transformer_predictions_mp.py ...
39358 S+     0.0  1.6 14:18:11  34:36.31 /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.spawn import spawn_main; ...
39359 S+     0.0  1.6 14:18:02  34:31.19 /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.spawn import spawn_main; ...
(remaining worker rows omitted for brevity; all in `S+` state, 0% CPU)
```

### Parent Stack Sample (`PID 35146`)

```bash
$ sample 35146 1
Analysis of sampling Python (pid 35146) every 1 millisecond
Process:         Python [35146]
Path:            /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python
...
    +                                     830 lock_PyThread_acquire_lock  (in Python) + 60
    +                                       830 _PyMutex_LockTimed  (in Python) + 464
    +                                         830 _PyParkingLot_Park  (in Python) + 312
    +                                           830 _PySemaphore_Wait  (in Python) + 108
    +                                             830 _pthread_cond_wait  (in libsystem_pthread.dylib) + 984
    +                                               830 __psynch_cvwait  (in libsystem_kernel.dylib) + 8
```

### Worker Stack Sample (`PID 39358`)

```bash
$ sample 39358 1
Analysis of sampling Python (pid 39358) every 1 millisecond
Process:         Python [39358]
Path:            /Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python
...
    +                       731 _multiprocessing_SemLock_acquire_impl  (in _multiprocessing.cpython-313-darwin.so) + 468
    +                         731 sem_wait  (in libsystem_kernel.dylib) + 8
```

### Output Progress Check

```bash
$ ls -lt nogit/finetune_results
total 26632
-rw-r--r--  1 ashutoshtiwari  staff  12811322 Oct 30 19:01 finetuned_1q_10g_results.jsonl

$ python3 -c "..."
records=20675
last_circuit_idx=20674

$ wc -l nogit_circuitdata/aug_angle_predictor_1022_start_subcircuits_gb10_20251028_093406_len10_enumall_100K_Nplus3.jsonl
59049 nogit_circuitdata/aug_angle_predictor_1022_start_subcircuits_gb10_20251028_093406_len10_enumall_100K_Nplus3.jsonl
```

### Diagnostic Report Excerpt

```bash
$ python3 -c "...ExcResource..."
termination: {'flags': 2, 'code': 14123288431434143079, 'namespace': 'PORT_SPACE', 'indicator': '(Limit 267623 ports) Exceeded system-wide per-process Port Limit'}
exception: {'codes': '0x0000000000041567, 0x0000000000000000', 'rawCodes': [267623, 0], 'type': 'EXC_RESOURCE', 'signal': 'SIGKILL'}
```

## Conclusions

- The pool parent thread is blocked on `lock_PyThread_acquire_lock`, awaiting an `imap` result that never arrives.
- Workers remain alive but are parked on `SemLock.acquire`, confirming no new tasks are dispatched.
- macOS `EXC_RESOURCE` diagnostics show repeated port-limit violations, implicating excessive Mach semaphore allocation from a huge backlog of pending jobs.

## Recommendations

- Feed the pool in manageable batches (e.g., split input list, or stream via generator that yields at most ~2 k jobs at a time).
- Provide a non-default `chunksize` (e.g., 32 or 64) so each task submission covers multiple circuits, reducing per-task semaphore usage.
- Enable `multiprocessing.util.log_to_stderr()` and add heartbeat logging/queue size metrics to detect feeder-thread failures promptly.
- Deploy the new watchdog script (`scripts/python_watchdog.py`) to track the process family’s CPU, memory, port count, and I/O while long jobs run.
