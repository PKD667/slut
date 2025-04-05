
// As a macro
#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr) => {
        assert_approx_eq!($left, $right, Scalar::EPSILON);
    };
    ($left:expr, $right:expr, $epsilon:expr) => {{
        let left_val = $left;
        let right_val = $right;
        let abs_diff = (left_val - right_val).norm();
        assert!(
            abs_diff.im() < $epsilon && abs_diff.re() < $epsilon,
            "assertion failed: `(left ≈ right)`\n  left: `{}`\n right: `{}`\n  diff: `{}`\n ",
            left_val, right_val, abs_diff,
        );
    }};
}

// macros for math shit
#[macro_export]
macro_rules! cvec {
    () => {
        // A 0-element vector; you may also choose to panic.
        Tensor::<crate::dimension::Dimensionless, 1, 0, 1>::zero()
    };
    ($($x:expr),+ $(,)?) => {{
        // Count the number of elements provided.
        const N: usize = <[()]>::len(&[$(cvec!(@replace $x)),*]);
        Tensor::<crate::dimension::Dimensionless, 1, N, 1>::new::<crate::units::Unitless>([
            $($x.complex()),*
        ])
    }};
    (@replace $_x:expr) => { () };
}

#[macro_export]
macro_rules! dless {
    ($x:expr) => {
        Tensor::<crate::dimension::Dimensionless, 1, 1, 1>::new::<crate::units::Unitless>([$x])
    };

}