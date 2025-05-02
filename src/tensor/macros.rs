
// filepath: /home/pkd/code/rust/dlt/src/tensor/macros.rs
#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr) => {{
        let left_val = $left;
        let epsilon = 1e-6;
        assert_approx_eq!(left_val, $right, epsilon);
    }};
    ($left:expr, $right:expr, $epsilon:expr) => {{
        let left_val = $left;
        let right_val = $right;
        let d = (ip!((left_val - right_val), (left_val - right_val))).item().mag().sqrt();
        
        if d > $epsilon {
            panic!(
                "assertion failed: `abs({:?} - {:?}) < {}`\n\
                left: {:?}, right: {:?}, distance: {}",
                stringify!($left),
                stringify!($right),
                $epsilon
                left_val,
                right_val,
                d
            );
        }
    }};
}


// macros for math shit
#[macro_export]
macro_rules! cvec {
    () => {
        // A 0-element vector; you may also choose to panic.
        Tensor::<dimension::Dimensionless, 1, 0, 1>::zero()
    };
    ($($x:expr),+ $(,)?) => {{
        // Count the number of elements provided.
        const N: usize = <[()]>::len(&[$(cvec!(@replace $x)),*]);
        Tensor::<c64,dimension::Dimensionless, 1, N, 1>::new::<units::Unitless>([
            $($x.complex()),*
        ])
    }};
    (@replace $_x:expr) => { () };
}

#[macro_export]
macro_rules! dless {
    ($x:expr) => {
        Tensor::<_, dimension::Dimensionless, 1, 1, 1>::new::<units::Unitless>([$x])
    };
}