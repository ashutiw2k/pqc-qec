from typing import Sequence, Any
from itertools import permutations, combinations


class SimpleCircuitTokenizer:
    """
    Tokenizes quantum circuit operations into unique integer IDs.

    This class creates a vocabulary of all possible 1-qubit and 2-qubit
    operations for a given gateset and number of qubits. It can then
    encode a circuit (a sequence of operations) into a list of token IDs
    and decode them back.

    Gate names are treated case-insensitively. A padding token ID can be
    specified.
    """

    def __init__(
        self,
        gateset: Sequence[str],
        num_qubits: int,
        qubits_for_gates: dict[str, int],
        undirected_gates: Sequence[str] | None = None,
    ):
        """
        Initializes the tokenizer and builds the operation vocabulary.

        Args:
            gateset: A sequence of supported gate names (e.g., ['h', 'cx']).
            num_qubits: The number of qubits in the device.
            undirected_gates: A sequence of 2-qubit gates that are
                symmetrical (e.g., 'cz'). For these, (gate, (q1, q2)) is
                the same as (gate, (q2, q1)).
            pad_id: The integer ID to reserve for padding.
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        self.num_qubits = num_qubits

        # Normalize gate names to lowercase for case-insensitive handling
        self.gates = {g.lower() for g in gateset}
        self.undirected = {g.lower() for g in (undirected_gates or [])}
        self.qubits_for_gates = {g.lower(): qubits_for_gates[g] for g in qubits_for_gates}

        # Validate that undirected gates are part of the main gateset
        if not self.undirected.issubset(self.gates):
            unknown = self.undirected - self.gates
            raise ValueError(f"Undirected gates {unknown} are not in the main gateset.")

        # --- Build Vocabulary ---
        # The vocabulary maps (gate_name, (qubits,)) tuples to integer IDs.
        # This is built once at initialization for efficiency.

        self.pad_id = 0
        # self.unk_id = 1

        self.pad_gate = "pad"
        # self.unk_gate = "unk"

        self.pad_token = (self.pad_gate, ())
        # self.unk_token = (self.unk_gate, ())

        self.op_to_id: dict[tuple[str, tuple[int, ...]], int] = {self.pad_token: self.pad_id}
        self.id_to_op: dict[int, tuple[str, tuple[int, ...]]] = {self.pad_id: self.pad_token}

        self._build_vocab()

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens, including padding."""
        return len(self.op_to_id)

    def _build_vocab(self) -> None:
        """
        Generates all 1- and 2-qubit operations using combinations/permutations
        and assigns deterministic IDs (no if-branches inside).
        """
        sorted_gates = sorted(self.gates)
        next_id = len(self.id_to_op)  # 0 is PAD
        qubits = range(self.num_qubits)
        # print(singles, directed_pairs, undirected_pairs)

        for g in sorted_gates:
            q_count = self.qubits_for_gates.get(g, None)
            qubit_tokens = list(combinations(qubits, q_count)) if g in self.undirected else list(permutations(qubits, q_count))
            for qs in qubit_tokens:
                op = (g, tuple(qs))
                self.op_to_id[op] = next_id
                self.id_to_op[next_id] = op
                next_id += 1

    def _canonicalize_op(self, gate: str, qubits: Sequence[int]) -> tuple[str, tuple[int, ...]]:
        """
        Normalizes an operation and validates its components.
        - Lowercases the gate name.
        - Sorts qubit indices for undirected gates.
        - Validates gate name, qubit count, and qubit indices.
        """
        g = gate.lower()
        if g not in self.gates:
            raise ValueError(f"Gate '{gate}' is not in the tokenizer's gateset.")

        qs = tuple(qubits)
        if not (1 <= len(qs) <= 2):
            raise ValueError(f"Only 1- and 2-qubit operations are supported, but got {len(qs)} qubits for gate '{gate}'.")
        
        for q in qs:
            if not (0 <= q < self.num_qubits):
                raise ValueError(f"Qubit index {q} is out of range for a device with {self.num_qubits} qubits.")

        if len(qs) == 2:
            if qs[0] == qs[1]:
                raise ValueError("A 2-qubit operation must act on two different qubits.")
            # For undirected gates, sort qubits to get the canonical form (e.g., (1, 2) not (2, 1))
            if g in self.undirected:
                qs = tuple(sorted(qs))
        
        return g, qs


    def encode(self, circuit: Sequence[tuple[str, Sequence[int], Any]]) -> list[int]:
        """
        Encodes a sequence of circuit operations into token IDs.

        Args:
            circuit: A list of operations, where each operation is a tuple
                     like ('gate_name', [qubit_indices], params). The params
                     are ignored.

        Returns:
            A list of integer token IDs corresponding to the operations.
        """
        token_ids = []
        for gate, qubits, *_ in circuit:
            op = self._canonicalize_op(gate, qubits)
            token_ids.append(self.op_to_id[op])
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> list[list[Any]]:
        """
        Decodes a sequence of token IDs back into circuit operations.

        Args:
            token_ids: A list of integer token IDs.

        Returns:
            A list of operations in the format [['gate_name', [qubits], []]].
            Padding tokens are ignored.
        """
        circuit = []
        for token_id in token_ids:
            if token_id == self.pad_id:
                continue
            
            if token_id not in self.id_to_op:
                raise KeyError(f"Token ID {token_id} is not in the vocabulary.")
            
            gate, qubits = self.id_to_op[token_id]
            circuit.append([gate, list(qubits), []])
        return circuit

    def print_vocab(self, limit: int | None = 20) -> None:
        """Print up to `limit` vocab entries by token id (ascending).
        Pass None to print all.
        """
        for token_id in sorted(self.id_to_op.keys())[:limit]:
            gate, qubits = self.id_to_op[token_id]
            print(f"ID {token_id:3}: Gate '{gate}' on qubits {qubits}")

    def __repr__(self) -> str:
        return (
            f"CircuitTokenizer(vocab_size={self.vocab_size}, num_qubits={self.num_qubits}, "
            f"pad_id={self.pad_id}, num_gates={len(self.gates)})"
        )
