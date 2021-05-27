//! Eigenvalue decomposition for Symmetric/Hermite matrices

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub trait Eigh_: Scalar {
    /// Wraps `*syev` for real and `*heev` for complex
    fn eigh(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    /// Wraps `*syevr` for real and `*heevr` for complex
    fn eigh_range(
        calc_v: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        range: Range<Self::Real>,
        abstol: Self::Real,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    /// Wraps `*syegv` for real and `*heegv` for complex
    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    (@real, $scalar:ty, $ev:path, $evs:path, $evg:path) => {
        impl_eigh!(@body, $scalar, $ev, $evs, $evg, );
    };
    (@complex, $scalar:ty, $ev:path, $evs:path, $evg:path) => {
        impl_eigh!(@body, $scalar, $ev, $evs, $evg, rwork);
    };
    (@body, $scalar:ty, $ev:path, $evs:path, $evg:path, $($rwork_ident:ident),*) => {
        impl Eigh_ for $scalar {
            fn eigh(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };

                $(
                let mut $rwork_ident = unsafe { vec_uninit(3 * n as usize - 2 as usize) };
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut work,
                        lwork as i32,
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(eigs)
            }

            fn eigh_range(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                range: Range<Self::Real>,
                abstol: Self::Real,
                mut a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };
                let (r, vl, vu, il, iu, num) = range.parameters();
                let (ldz, ndz) = if jobz == b'V' {
                    match num {
                        Some(x) => (n, x),
                        None => (n,n)
                    }
                } else {
                    (1, 1)
                };
                let mut z = unsafe { vec_uninit((ldz*ndz) as usize) };
                let mut isuppz = unsafe { vec_uninit((2*ndz) as usize) };


                // calc work size
                let mut info = 0;
                let mut n_eigs = 0;
                let mut work_size = [Self::zero()];
                let mut iwork_size = [0];
                $(
                let mut $rwork_ident = [Self::Real::zero()];
                )*
                unsafe {
                    $evs(
                        jobz,
                        r,
                        uplo as u8,
                        n,
                        &mut a,
                        n, // lda
                        vl,vu,il,iu,abstol,
                        &mut n_eigs, // m
                        &mut eigs, // w
                        &mut z,
                        ldz, // ldz
                        &mut isuppz,
                        &mut work_size, // work
                        -1, // lwork
                        $(&mut $rwork_ident,-1,)* // rwork, lrwork
                        &mut iwork_size, // iwork
                        -1, // liwork
                        &mut info, // info
                    );
                }
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                let liwork = iwork_size[0].to_usize().unwrap();
                let mut iwork = unsafe { vec_uninit(liwork) };
                $(
                let lrwork = $rwork_ident[0].to_usize().unwrap();
                let mut $rwork_ident = unsafe { vec_uninit(lrwork) };
                )*
                unsafe {
                    $evs(
                        jobz,
                        r,
                        uplo as u8,
                        n,
                        &mut a,
                        n, // lda
                        vl,vu,il,iu,abstol,
                        &mut n_eigs, // m
                        &mut eigs, // w
                        &mut z,
                        ldz, // ldz
                        &mut isuppz,
                        &mut work, // work
                        lwork as i32, // lwork
                        $(&mut $rwork_ident, lrwork as i32,)* // rwork, lrwork
                        &mut iwork, // iwork
                        liwork as i32, // liwork
                        &mut info, // info
                    );
                }
                info.as_lapack_result()?;
                for i in 0..z.len() {
                    a[i] = z[i];
                }
                Ok(eigs[0..n_eigs as usize].to_vec())
            }

            fn eigh_generalized(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
                mut b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };

                $(
                let mut $rwork_ident = unsafe { vec_uninit(3 * n as usize - 2) };
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $evg(
                        &[1],
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual evg
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    $evg(
                        &[1],
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut eigs,
                        &mut work,
                        lwork as i32,
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(eigs)
            }
        }
    };
} // impl_eigh!

impl_eigh!(@real, f64, lapack::dsyev, lapack::dsyevr, lapack::dsygv);
impl_eigh!(@real, f32, lapack::ssyev, lapack::ssyevr, lapack::ssygv);
impl_eigh!(@complex, c64, lapack::zheev, lapack::zheevr, lapack::zhegv);
impl_eigh!(@complex, c32, lapack::cheev, lapack::cheevr, lapack::chegv);
