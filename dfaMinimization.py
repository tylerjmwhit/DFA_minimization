# Class to hold DFA
import timeit

## Tyler Whitmarsh
## Jeremy Pearson

class DFA(object):
    def __init__(self, States, Symbols, Delta, Start, Final):
        """
        Constructs a new DFA object with the given set of states, input symbols, 
        delta function, initial state, and set of final states.
        
        Parameters:
        States (set): A set of states in the DFA.
        Symbols (list): A list of input symbols in the DFA.
        Delta (dict): A dictionary representing the delta function. 
                      The keys are tuples of the form (state, symbol) and the 
                      values are the corresponding next states.
        Start: The initial state of the DFA.
        Final (set): A set of final states in the DFA.
        """
        self.States = States
        self.Symbols = Symbols
        self.Delta = Delta
        self.Start = Start
        self.Final = Final
    
    def copy(self):
        """
        Returns a copy of the DFA object.
        
        Returns:
        DFA: A new DFA object with the same attributes as the original.
        """
        return DFA(
            States = set(self.States),
            Symbols = list(self.Symbols),
            Delta = {state: dict(Delta) for state, Delta in self.Delta.items()},
            Start = self.Start,
            Final = set(self.Final)
        )
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the DFA object.
        
        Returns:
        str: A string representation of the DFA object.
        """
        return f"DFA(states={self.States}, input_symbols={self.Symbols}, " \
            f"transitions={self.Delta}, initial_state='{self.Start}', " \
            f"final_states={self.Final})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the DFA object.
        
        Returns:
        str: A string representation of the DFA object.
        """

        delta_str = "{\n"
        for state, transitions in self.Delta.items():
            delta_str += f"    '{state}': {transitions},\n"
        delta_str += "}"

        return f"\nDFA" \
            f"\nStates {self.States}" \
            f"\nSymbols  {self.Symbols}" \
            f"\nDelta  {delta_str}"  \
            f"\nStart  {self.Start}"  \
            f"\nFinal  {self.Final}" 
    
# Unreachable states
    def remove_unreachable(DFA):
        """
        Returns a new DFA that is equivalent to the current DFA but with all the unreachable
        states removed.

        Parameters:
            DFA (DFA): The input DFA (Deterministic Finite Automaton) object.

        Returns:
            DFA: A new DFA object equivalent to the input DFA, but with unreachable states removed.
        """
        reachable = set()
        visted_stack = [DFA.Start]
        while visted_stack:
            current = visted_stack.pop()
            if current not in reachable:
                reachable.add(current)
                for symbol, next in DFA.Delta[current].items():
                    visted_stack.append(next)
        
        new_dfa = DFA.copy()
        new_dfa.States = reachable
        new_dfa.Start = DFA.Start
        new_dfa.Final = DFA.Final.intersection(reachable)
        new_dfa.Delta = {
            state: {symbol: next for symbol, next in delta.items() if next in reachable} 
            for state, delta in DFA.Delta.items()
            if state in reachable
        }
        return new_dfa

#Nondistinguishable states
    def remove_nondistinguishable_states(DFA, pre = True):
        """
        Returns a new DFA that is equivalent to the current DFA but with all the non-distinguishable
        states removed.

        This algorithm works by first removing unreachable states and then merging states that cannot
        be distinguished by any input symbol.

        Parameters:
            DFA (DFA): The input DFA (Deterministic Finite Automaton) object.
            pre (bool): default true: Input bool to determine if remove_unreachable should be done first

        Returns:
            DFA: A new DFA object equivalent to the input DFA, but with non-distinguishable states removed.
        """
        if pre:
            DFA = DFA.remove_unreachable()
        p0 = [DFA.Final , DFA.States - DFA.Final]
        p1 = [DFA.Final , DFA.States - DFA.Final]

        while p1:
            A = p1.pop(0)

            for C in DFA.Symbols:

                X = { state for state in DFA.States if DFA.Delta[state][C] in A}

                for Y in p0:
                    if X.intersection(Y) and Y.difference(X):
                        p0.remove(Y)
                        p0.append(X.intersection(Y))
                        p0.append(Y.difference(X))

                        if Y in p1:
                            p1.remove(Y)
                            p1.append(X.intersection(Y))
                            p1.append(Y.difference(X))
                        else:
                            if len(X.intersection(Y)) <= len(Y.difference(X)):
                                p1.append(X.intersection(Y))
                            else:
                                p1.append(Y.difference(X))


        state_to_partition = {}
        for i, partition in enumerate(p0):
            for state in partition:
                state_to_partition[state] = partition

        states = []
        delta = {}
        start_state = None
        accepting_states = []

        for i, P_i in enumerate(p0):
            if DFA.Start in P_i:
                start_state = P_i

            if P_i.intersection(DFA.Final):
                accepting_states.append(P_i)

            states.append(P_i)

            temp_delta = {}
            for c in DFA.Symbols:
                next_states = {DFA.Delta[state][c] for state in P_i}
                next_partition = state_to_partition[next_states.pop()]  # Retrieve the partition of the next state

                temp_delta[c] = next_partition

            delta[tuple(P_i)] = temp_delta

        new_dfa = DFA.copy()
        new_dfa.States = states
        new_dfa.Start = start_state
        new_dfa.Final = accepting_states
        new_dfa.Delta = delta
        return new_dfa

       
def main():
    dfa1 = DFA(
    States={'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'},
    Symbols=['0', '1'],
    Delta={
        'q0': {'0': 'q3', '1': 'q1'},
        'q1': {'0': 'q2', '1': 'q5'},
        'q2': {'0': 'q2', '1': 'q5'},
        'q3': {'0': 'q0', '1': 'q4'},
        'q4': {'0': 'q2', '1': 'q5'},
        'q5': {'0': 'q5', '1': 'q5'},
        'q6': {'0': 'q3', '1': 'q1'},
    },
    Start='q0',
    Final={'q1' ,'q2' , 'q4'}
    )

    dfa2 = DFA(
    States={'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6',
            'q7','q8','q9','q10','q11','q12','q13','q14',
            'q15'},
    Symbols=['0', '1'],
    Delta={
        'q0': {'0': 'q3', '1': 'q1'},
        'q1': {'0': 'q2', '1': 'q5'},
        'q2': {'0': 'q2', '1': 'q5'},
        'q3': {'0': 'q0', '1': 'q4'},
        'q4': {'0': 'q2', '1': 'q5'},
        'q5': {'0': 'q5', '1': 'q5'},
        'q6': {'0': 'q3', '1': 'q1'},
        'q7': {'0': 'q3', '1': 'q1'},
        'q8': {'0': 'q3', '1': 'q1'},
        'q9': {'0': 'q3', '1': 'q1'},
        'q10': {'0': 'q3', '1': 'q1'},
        'q11': {'0': 'q3', '1': 'q1'},
        'q12': {'0': 'q3', '1': 'q1'},
        'q13': {'0': 'q3', '1': 'q1'},
        'q14': {'0': 'q3', '1': 'q1'},
        'q15': {'0': 'q3', '1': 'q1'},
    },
    Start='q0',
    Final={'q1' ,'q2' , 'q4'}
    )

    dfa3 = DFA(
    States={'q0', 'q1', 'q2', 'q3', 'q4'},
    Symbols=['0', '1'],
    Delta={
        'q0': {'0': 'q1', '1': 'q4'},
        'q1': {'0': 'q3', '1': 'q0'},
        'q2': {'0': 'q3', '1': 'q0'},
        'q3': {'0': 'q3', '1': 'q0'},
        'q4': {'0': 'q3', '1': 'q4'}
    },
    Start='q0',
    Final={'q4'}
    )

    print("\ndfa 1 \n")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_unreachable(dfa1)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_nondistinguishable_states(dfa1)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_nondistinguishable_states(dfa1, False)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")

    print("\ndfa 2 \n")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_unreachable(dfa2)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_nondistinguishable_states(dfa2)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_nondistinguishable_states(dfa2, False)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    
    print("\ndfa 3 \n")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_unreachable(dfa3)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_nondistinguishable_states(dfa3)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")
    result =timeit.timeit(stmt= 'lambda: DFA.remove_nondistinguishable_states(dfa3, False)', globals=globals(), number = 1000)
    print(f"{result / 1000} seconds")

    print("Current DFA is", dfa1)
    new_dfa = DFA.remove_unreachable(dfa1)
    print("\nAfter removing unreachable states DFA is" , new_dfa)
    minimized_dfa = DFA.remove_nondistinguishable_states(dfa1)
    print("\nHopcroft minimized DFA is" , minimized_dfa)


    print("\nCurrent DFA is", dfa2)
    new_dfa = DFA.remove_unreachable(dfa2)
    print("\nAfter removing unreachable states DFA is" , new_dfa)
    minimized_dfa = DFA.remove_nondistinguishable_states(dfa2)
    print("\nHopcroft minimized DFA is" , minimized_dfa)



    print("\nCurrent DFA is", dfa3)
    new_dfa = DFA.remove_unreachable(dfa3)
    print("\nAfter removing unreachable states DFA is" , new_dfa)
    minimized_dfa = DFA.remove_nondistinguishable_states(dfa3)
    print("\nHopcroft minimized DFA is" , minimized_dfa)

if __name__ == '__main__':
    main()


