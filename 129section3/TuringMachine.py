class TuringMachine:
    def __init__(self, A, B):
        self.tape = list(f"{A}#{B}#0")
        self.head = 0
        self.state = "init"
        self.steps = []
        self.step_count = 0 
        self.max_steps = 100000  

    def transition(self):
        while self.state != "halt" and self.step_count < self.max_steps:
            self.steps.append("".join(self.tape))
            self.step_count += 1  
        return self.steps

def save_tape(A, B):
    tm = TuringMachine(A, B)
    steps = tm.transition()
    filename = f"{A}_{B}.dat"
    with open(filename, "w") as f:
        f.write("\n".join(steps))
if __name__ == "__main__":
    save_tape("101001010111", "101000101")
    save_tape("101111", "101001")

