use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use numpy::ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;

/// Compute thin SVD of a 2D f32 array using faer.
///
/// Returns (S, Vh) where:
///   S  has shape (k,)    — singular values in descending order
///   Vh has shape (k, n)  — right singular vectors as rows
/// with k = min(m, n).
///
/// Pass a Fortran-order (column-major) array for zero-copy ingestion.
/// C-order arrays are transposed with a cache-efficient blocked copy.
#[pyfunction]
fn svd_thin<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
    let arr = array.as_array();
    let m = arr.nrows();
    let n = arr.ncols();
    let k = m.min(n);

    let svd = if !arr.is_standard_layout() {
        // F-contiguous (column-major): hand directly to faer — zero copy.
        let slice = arr.as_slice_memory_order()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Array is not contiguous"))?;
        faer::mat::MatRef::<f32>::from_column_major_slice(slice, m, n)
            .thin_svd()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e:?}")))?
    } else {
        // C-contiguous (row-major): blocked transpose into column-major.
        // Each 32×32 block (2 × 32×32×4 = 8 KB) fits in L1 cache.
        let slice = arr.as_slice()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Array is not C-contiguous"))?;
        let mut col_major = vec![0.0f32; m * n];
        const BLOCK: usize = 32;
        for jj in (0..n).step_by(BLOCK) {
            let jmax = (jj + BLOCK).min(n);
            for ii in (0..m).step_by(BLOCK) {
                let imax = (ii + BLOCK).min(m);
                for j in jj..jmax {
                    for i in ii..imax {
                        col_major[j * m + i] = slice[i * n + j];
                    }
                }
            }
        }
        faer::mat::MatRef::<f32>::from_column_major_slice(&col_major, m, n)
            .thin_svd()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{e:?}")))?
    };

    let s_col = svd.S().column_vector();
    let v_mat = svd.V();

    let s_vec: Vec<f32> = (0..k).map(|i| *s_col.get(i)).collect();
    let vh_flat: Vec<f32> = (0..k)
        .flat_map(|i| (0..n).map(move |j| *v_mat.get(j, i)))
        .collect();
    let vh_nd = Array2::from_shape_vec((k, n), vh_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        s_vec.into_pyarray(py),
        vh_nd.into_pyarray(py),
    ))
}

/// Sample sequences from a position weight matrix (inverse CDF / multinomial sampling).
///
/// Equivalent to tpcav.utils.sample_from_pwm.
///   pwm_prob_mat: (L, A) float64 probability matrix; rows must sum to ~1.
///   n_seqs: number of sequences to generate (default 1).
///   seed: optional u64 RNG seed for reproducibility.
///   alphabet: optional list of single-char strings (default ['A','C','G','T']).
/// Returns a list of n_seqs strings, each of length L.
///
/// Implementation: each sequence is processed independently using a per-sequence
/// SmallRng (Xoshiro256++) seeded via multiplicative hashing of the base seed,
/// so all sequences can be generated in parallel with rayon.
#[pyfunction]
#[pyo3(signature = (pwm_prob_mat, n_seqs=1, seed=None, alphabet=None))]
fn sample_from_pwm(
    pwm_prob_mat: PyReadonlyArray2<f64>,
    n_seqs: usize,
    seed: Option<u64>,
    alphabet: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let arr = pwm_prob_mat.as_array();
    let l = arr.nrows();
    let a = arr.ncols();

    let alpha_bytes: Vec<u8> = match alphabet {
        Some(ref v) => {
            if v.len() != a {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("alphabet length {} != PWM columns {}", v.len(), a),
                ));
            }
            v.iter()
                .map(|s| {
                    s.as_bytes().first().copied().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("empty alphabet entry")
                    })
                })
                .collect::<PyResult<Vec<u8>>>()?
        }
        None => b"ACGT".to_vec(),
    };

    // Flat (l × a) cumulative sums, row-major: cum_flat[pos*a .. (pos+1)*a].
    let mut cum_flat = Vec::with_capacity(l * a);
    for i in 0..l {
        let mut running = 0.0_f64;
        for j in 0..a {
            running += arr[[i, j]];
            cum_flat.push(running);
        }
    }

    // Derive base seed; if none given, sample one from the thread-local RNG.
    let base_seed: u64 = seed.unwrap_or_else(|| rand::rng().random());

    // Each sequence gets a unique SmallRng (Xoshiro256++) seeded by mixing
    // base_seed with the sequence index, so all sequences are independent
    // and can be processed in parallel.
    let seqs: Vec<String> = (0..n_seqs)
        .into_par_iter()
        .map(|si| {
            let seq_seed = base_seed ^ (si as u64).wrapping_mul(0x9e3779b97f4a7c15u64);
            let mut rng = SmallRng::seed_from_u64(seq_seed);
            let mut bytes = Vec::with_capacity(l);
            for pos in 0..l {
                let u: f64 = rng.random();
                let row = &cum_flat[pos * a..(pos + 1) * a];
                let idx = row.iter().position(|&c| c > u).unwrap_or(a - 1);
                bytes.push(alpha_bytes[idx]);
            }
            // SAFETY: alpha_bytes contains only ASCII characters.
            unsafe { String::from_utf8_unchecked(bytes) }
        })
        .collect();

    Ok(seqs)
}

#[pymodule]
fn rust_optim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(svd_thin, m)?)?;
    m.add_function(wrap_pyfunction!(sample_from_pwm, m)?)?;
    Ok(())
}
