class NondeterministicTuringMachine:
    def __init__(self, A, B):
        self.tape = list(f"{A}#{B}#0")
        self.head = 0
        self.state = "choose"
        self.step_count = 0
        self.max_steps = 10_000  # Prevent infinite loops

    def transition(self):
        while self.state != "halt" and self.step_count < self.max_steps:
            current_symbol = self.tape[self.head]
            
            if self.state == "choose":
                # Nondeterministically choose to add or skip
                choice = np.random.choice(["add", "skip"])
                if choice == "add":
                    self.state = "add"
                else:
                    self.state = "skip"
            
            elif self.state == "add":
                self.tape[self.head] = 'X'  # Mark bit
                self.state = "shift"
                self.head += 1
            
            elif self.state == "skip":
                self.head += 1
                self.state = "choose"
            
            elif self.state == "shift":
                # Shift logic (symmetric for A and B)
                self.head -= 1
                self.state = "choose"
            
            self.step_count += 1
        return self.step_count
