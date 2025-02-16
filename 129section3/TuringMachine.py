class TuringMachine:
    def __init__(self, A, B):
        self.tape = list(f"{A}#{B}#0")
        self.head = 0
        self.state = "init"
        self.step_count = 0
        self.max_steps = 100_000  # Prevent infinite loops

    def transition(self):
        """Execute state transitions until halt or max_steps."""
        while self.state != "halt" and self.step_count < self.max_steps:
            # Ensure head is within tape bounds (expand tape if needed)
            if self.head < 0:
                self.head = 0  # Reset to start
            elif self.head >= len(self.tape):
                self.tape.append('0')  # Expand tape to the right

            current_symbol = self.tape[self.head]

            # State transition logic (example)
            if self.state == "init":
                if current_symbol == '#':
                    self.state = "read_B"
                self.head += 1

            elif self.state == "read_B":
                if current_symbol == '1':
                    self.state = "add_A"
                elif current_symbol == '#':
                    self.state = "halt"  # End of B
                self.head += 1

            elif self.state == "add_A":
                self.tape[self.head] = 'X'  # Mark processed bit
                self.state = "shift_A"
                self.head += 1

            elif self.state == "shift_A":
                self.head -= 1  # Move back to A
                if self.head < 0:  # Prevent underflow
                    self.head = 0
                self.state = "read_B"

            self.step_count += 1

        return self.step_count

if __name__ == "__main__":
    save_tape("101001010111", "101000101")
    save_tape("101111", "101001")

