/// A way of chaining "Thingies" (i lack a better name) together
#[derive(Debug)]
pub enum Operation<'a> {
    /// a + b
    Add(Box<Thingy<'a>>, Box<Thingy<'a>>),
    /// a * b
    Multiply(Box<Thingy<'a>>, Box<Thingy<'a>>),
    // /// a^b
    // Power(Thingy, Thingy),
}

#[derive(Debug)]
pub enum Thingy<'a> {
    /// A fixed floating point number (eg. 2.5 or -1)
    Scalar(f32),
    /// a sub-operation (something like "(1+2) * 3" )
    Operation(Operation<'a>),
    /// A floating point number whose value will be adjusted when backpropagating
    MutableScalar(&'a mut f32),
}

// /// two Thingies being chained together by an operator eg. "1+2" or "3^2"
// pub struct Expression {
//     a: Box<Thingy>,
//     op: Operator,
//     b: Box<Thingy>,
// }

impl Operation<'_> {
    pub fn backprop(&self, grad: f32) {
        match self {
            Operation::Add(a, b) => {
                match **a {
                    Thingy::Operation(ref op_a) => op_a.backprop(grad),
                    Thingy::Scalar(_) => {}, // can't be modified => no backprop can be done
                    Thingy::MutableScalar(ref mut_scalar) => {}, // dunno what to do here yet
                }
                match **b {
                    Thingy::Operation(ref op_b) => op_b.backprop(grad),
                    Thingy::Scalar(_) => {}, // can't be modified => no backprop can be done
                    Thingy::MutableScalar(ref mut_scalar) => {}, // dunno what to do here yet
                }
            }

            Operation::Multiply(a, b) => {
            },
        }
    }

    /// get the mathematical result of that operation
    fn value(&self) -> f32 {
        match self {
            Operation::Add(a, b) => a.value() + b.value(),
            Operation::Multiply(a, b) => a.value() * b.value(),
        }
    }
}

impl Thingy<'_> {
    /// get the mathematical value of that Thingy
    fn value(&self) -> f32 {
        match self {
            Thingy::Operation(op) => op.value(),
            Thingy::Scalar(s) => *s,
            Thingy::MutableScalar(&mut s) => s,
        }
    }
}
