# Static Linear Unitfull Tensors

This is a small lib I made for my thermodynamics simulation project. I was fed up with implementing vectors in `uom`. It's very bug prone, experimental but does the job quite well. Inspired by [Terry Tao's post](https://terrytao.wordpress.com/2012/12/29/a-mathematical-formalisation-of-dimensional-analysis/#xml) and [yuouom](https://github.com/iliekturtles/uom).  

It's very useful for a lot of things like physics simulation, linear algebra, and could even be used as a theorem prover for dimensional analysis. Don't forget to include the experimental headers and use cargo nightly.

```rust 
#![feature(generic_const_exprs)]
#![feature(trivial_bounds)]
#![feature(generic_arg_infer)]
```

## Usage

You can cargo add the library from this repository. You can find all the structures in `dlt::tensor` units in `dlt::units` and all the dimensions in `dlt::dimension`. You can create tensors of any dimension and any unit. You can also perform operations on tensors like addition, subtraction, scaling, dot product, cross product, matrix multiplication, etc.

```rust
use dlt::tensor::*;
use dlt::dimension::*;
use dlt::units::*;

let t = Tensor::<Length, 2, 1>::new::<f32,Meter>([1.0, 2.0]);
let v = Vec2::<f32,Force>::new::<Lbf>([1.0, 2.0]);
let m = Mat3::<f32,Force>::new::<Lbf>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
```

## Example
```rust

        use dlt::tensor::*;
        use dlt::dimension::*;
        use dlt::units::*;

        // Tensor of lengths
        let mass = Scalar::<f32,Mass>::new::<Kilogram>([10.0]);
        let force = Vec2::<f32,Force>::new::<Newton>([10.0, 20.0]);

        //let error = mass + force; // error (expected)

        let mass = mass + Scalar::<f32,Mass>::new::<Gram>([5.0]); // works
        println!("{}", mass);
        /*
        Tensor [1x1]: M^1
        ( 10.005 )
        */

        // Operator overloading for scalar division
        let acc = force / mass; // works
        println!("{}", acc);
        /*
        Tensor [2x1]: L^1 * T^-2
        ( 0.9995002 )
        ( 1.9990004 )
        */

        let time = Scalar::<f32,Time>::from::<Second>(1.0);
        let vel1 = Vec2::<f32,Velocity>::new::<MetersPerSecond>([10.0, 20.0]);

        let vel2 = vel1 + acc.scale(time); // works
        println!("{}", vel2.get::<f32,MetersPerSecond>());
        /*
        [10.9995, 21.999]
        */
```

## Linear algebra 

```rust

        // try to transpose a tensor
        let tensor = Tensor::<f64,Dimensionless, 1, 6>::new::<Unitless>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor_transposed = tensor.transpose();
        println!("{}", tensor);
        println!("{}", tensor_transposed);
        /*
        Tensor [1x6]: Dimensionless
        ( 1  2  3  4  5  6 )

        Tensor [6x1]: Dimensionless
        ( 1 )
        ( 2 )
        ( 3 )
        ( 4 )
        ( 5 )
        ( 6 )
         */

        let length = Vec2::<f32,Length>::new::<Meter>([1.0, 2.0]);

        // now try and dot product length and force
        // dot product add the dimensions
        let dot_product = dot!(length, force);
        println!("{}", dot_product);
        /*
        Tensor [1x1]: L^2 * M^1 * T^-2
        ( 50 )
        */

        assert_dimension!(dot_product, Energy); // works
        //assert_dimension!(dot_product, Force); // error (expected)

        let m1 = Matrix::<f32,Length, 2, 3>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Matrix::<f32,Length, 3, 2>::new::<Meter>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let m3 = m1.matmul(m2);
        println!("{}", m3);
        /*
        Tensor [2x2]: L^2
        ( 22  28 )
        ( 49  64 )
        */
```
