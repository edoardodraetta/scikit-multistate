"""
Module for the generation of state diagrams for multistate models.
"""

import base64
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from IPython.display import Image, display


def state_diagram(graph):
    """Plot a state diagram for a graph.

    See http://mermaid-js.github.io/mermaid/#/Tutorials?id=jupyter-integration-with-mermaid-js

    Example:
        state_diagram(
            '''stateDiagram-v2
            s1 : (1) Primary surgery
            s2: (2) Disease recurrence
            s3: (3) Death
            s1 --> s2
            s1 --> s3
            s2 --> s3
            ''')
    """
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    img = Image(url="https://mermaid.ink/img/" + base64_string)
    display(img)


class StateDiagramGenerator:
    """
    Generate state transition diagrams from multistate model data.

    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        patient_id: str,
        from_state: str,
        to_state: str,
        tstart: str,
        tstop: str,
        status: str,
        state_labels: Dict[int, str] | None = None,
        terminal_states: List[int] | None = None,
    ):
        """Initialize the generator.

        Args:
            dataset (pd.DataFrame): Transition dataset in the proper format.
            patient_id (str): Patient id column.
            from_state (str): From state column.
            to_state (str): To state column.
            tstart (str): Start time in from state.
            tstop (str): Stop time in from state.
            status (str): Whether or not the transition to to state has occurred.
            state_labels (str, optional): Optional state labels. Defaults to None.
            terminal_states (str, optional): Definition of terminal states. Defaults to None.
        """
        self.state_labels = state_labels or {}
        self.terminal_states = set(terminal_states or [])
        self.states = set()
        self.transitions = defaultdict(list)
        self.transition_counts = defaultdict(int)
        self.transition_times = defaultdict(list)

        # column names
        self.patient_id = patient_id
        self.from_state = from_state
        self.to_state = to_state
        self.tstart = tstart
        self.tstop = tstop
        self.status = status

        # Load data
        self.data = self._load_data(dataset)

    def _load_data(self, data: pd.DataFrame, validate: bool = True) -> None:
        """
        Load multistate event history data.

        Parameters
        ----------
        data : pd.DataFrame
        validate : bool, default True
            Whether to validate the data format
        """
        if validate:
            self._validate_data(data)
        self.data = data.copy()
        self._process_transitions()

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate the input data format.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to validate

        Raises
        ------
        ValueError
            If data format is invalid
        """
        required_columns = [
            self.patient_id,
            self.from_state,
            self.to_state,
            self.tstart,
            self.tstop,
            self.status,
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for negative time values
        if (data[self.tstart] < 0).any() or (data[self.tstop] < 0).any():
            raise ValueError("Time values cannot be negative")

        # Check that tstop >= tstart
        if (data[self.tstop] < data[self.tstart]).any():
            raise ValueError("tstop must be >= tstart")

        # Check status values are 0 or 1
        if not data[self.status].isin([0, 1]).all():
            raise ValueError("Status must be 0 (censored) or 1 (event)")

    def _process_transitions(self) -> None:
        """
        Process the loaded data to extract state transitions and statistics.
        """
        # Reset collections
        self.transitions = defaultdict(list)
        self.transition_counts = defaultdict(int)
        self.transition_times = defaultdict(list)
        self.states = set()

        if self.data is not None:
            # Group by patient to track individual trajectories
            for _, patient_data in self.data.groupby(self.patient_id):
                # Sort by time to ensure proper sequence
                patient_data = patient_data.sort_values([self.tstart, self.tstop])

                for _, row in patient_data.iterrows():
                    origin = int(row[self.from_state])
                    target = int(row[self.to_state])
                    status = int(row[self.status])
                    transition_time = row[self.tstop] - row[self.tstart]

                    # Add states to our set
                    self.states.add(origin)
                    self.states.add(target)

                    # Only count actual transitions (status = 1)
                    if status == 1:
                        transition_key = (origin, target)
                        self.transitions[origin].append(target)
                        self.transition_counts[transition_key] += 1
                        self.transition_times[transition_key].append(transition_time)

    def add_state_labels(self, labels: Dict[int, str]) -> None:
        """
        Add or update state labels.

        Parameters
        ----------
        labels : dict
            Mapping from state IDs to human-readable labels
        """
        self.state_labels.update(labels)

    def set_terminal_states(self, terminal_states: List[int]) -> None:
        """
        Set which states are terminal (absorbing).

        Parameters
        ----------
        terminal_states : list
            List of terminal state IDs
        """
        self.terminal_states = set(terminal_states)

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get transition count matrix.

        Returns
        -------
        pd.DataFrame
            Matrix showing transition counts between states
        """
        states = sorted(self.states)
        matrix = pd.DataFrame(0, index=states, columns=states)

        for (origin, target), count in self.transition_counts.items():
            matrix.loc[origin, target] = count

        return matrix

    def extract_state_diagram_string_from_transition_table(self) -> str:
        """This function extracts a mermaid state diagram string"""

        graph = """stateDiagram-v2\n"""

        # Add state labels
        for s in sorted(self.states):
            state_label = self.state_labels.get(s, f"State {s}")
            graph += f"""s{s} : ({s}) {state_label}\n"""

        # Add transitions
        for (origin_state, target_state), count in self.transition_counts.items():
            if count > 0:  # Only include actual transitions
                graph += f"""s{origin_state} --> s{target_state}: {count} \n"""

        # Add terminal state transitions
        for state in self.terminal_states:
            if state in self.states:
                graph += f"""s{state} --> [*]\n"""

        graph += """\n"""
        self.state_diagram_graph_string = graph
        return graph

    def plot_state_diagram(self):
        """This function plots a mermaid state diagram for the model"""
        if not hasattr(self, "state_diagram_graph_string") or self.state_diagram_graph_string is None:
            self.extract_state_diagram_string_from_transition_table()
        return state_diagram(self.state_diagram_graph_string)
