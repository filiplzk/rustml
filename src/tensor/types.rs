use std::fmt::{Debug, Display};
use std::ops::{Add, Sub, Mul, Div};
use num_traits::{Float, Num, NumAssignOps, NumCast, NumOps};
use rand::distr::uniform::SampleUniform;

pub trait AnyNumber:
    Num
    + NumCast
    + NumAssignOps
    + PartialOrd
    + Copy
    + Debug
    + Display
{
}

impl AnyNumber for i8 {}
impl AnyNumber for i16 {}
impl AnyNumber for i32 {}
impl AnyNumber for i64 {}
impl AnyNumber for i128 {}
impl AnyNumber for isize {}
impl AnyNumber for u8 {}
impl AnyNumber for u16 {}
impl AnyNumber for u32 {}
impl AnyNumber for u64 {}
impl AnyNumber for u128 {}
impl AnyNumber for usize {}
impl AnyNumber for f32 {}
impl AnyNumber for f64 {}


pub trait AnyFloat:
    AnyNumber
    + Float
{
}

impl AnyFloat for f32 {}
impl AnyFloat for f64 {}