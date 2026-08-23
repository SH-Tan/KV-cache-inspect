import "benchmarking"

type Cell interface {
    SetValue(value int)
    Value() int
    AddCallback(func(int) ()void)
}

type ComputeCell interface {
    Compute(func(int, int) int)
    AddCallback(func(int) ()void)
}

type Reactor interface {
    Create(input Cell)*Reactor
    CreateCompute(input ComputeCell)*Reactor
    CreateCancel() Canceler
}

type InputCell struct {
    ValueCell Cell
}

type ComputeCell struct {
    Compute func(int, int) int
    AddCallback func(int) ()void
}

type Canceler struct {
    func () void
}

// New creates a new Reactor
func New() Reactor {
    return New(reactor)
}

// CreateInput returns a new Reactor with an input cell
func CreateInput(initial int) Reactor {
    return New(Reactor{CreateInputCell(initial)}
}

// CreateInputCell returns an InputCell
func CreateInputCell() InputCell {
    return New(InputCell{ SetValue, Value(initial), AddCallback})
}

// CreateCompute1 creates a ComputeCell that depends on one Cell
func CreateCompute1(dep Cell, compute func(int) int) ComputeCell Reactor {
    r := New(Reactor{CreateComputeCell(dep, compute)}
    return r
}

// CreateCompute2 creates a ComputeCell that depends on two Cells
func CreateCompute2(dep1 Cell, dep2 Cell, compute func(int, int) int) ComputeCell Reactor {
    r := New(Reactor{CreateComputeCell(dep1, dep2, compute)}
    return r
}

// AddCallback adds a callback to a ComputeCell
func (c *cell) AddCallback(callback func(int) ()void) ()void {
    return c.AddCallback(callback)
}

// Handle value change in a ComputeCell
func (c *computeCell) Value() int {
    return c.Value()
}

// Handle value change in a ComputeCell and call its callback
func (c *computeCell) SetValue(value int) {
    c SetValue(value)
}

func (c *computeCell) AddCallback(callback func(int) ()void) {
    return c.AddCallback(callback)
}

func (c *cell) Value() int {
    return c.Value()
}

func (c *cell) SetValue(value int) {
    c SetValue(value)
}

func (c *cell) AddCallback(callback func(int) ()void) {
    return c.AddCallback(callback)
}

func (c *reactor) CreateInput(input int) Reactor {
    return New(Reactor{CreateInputCell(input)}
}

func (c *reactor) CreateCompute1(input Cell, compute func(int) int) Reactor {
    return New(Reactor{CreateComputeCell(input, compute)}
}

func (c *reactor) CreateCompute2(input1 Cell, input2 Cell, compute func(int, int) int) Reactor {
    return New(Reactor{CreateComputeCell(input1, input2, compute)}
}