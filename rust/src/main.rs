fn main() {
    // 4x3 test matrix with known rank-2 structure
    #[rustfmt::skip]
    let data: &[&[f64]] = &[
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0],
        &[10.0, 11.0, 12.0],
    ];
    let m = data.len();
    let n = data[0].len();
    let k = m.min(n);

    let mat = faer::Mat::<f64>::from_fn(m, n, |i, j| data[i][j]);

    println!("Input matrix ({m}x{n}):");
    for i in 0..m {
        let row: Vec<String> = (0..n).map(|j| format!("{:8.3}", *mat.get(i, j))).collect();
        println!("  {}", row.join(""));
    }

    // Thin SVD
    let svd = mat.thin_svd().expect("SVD failed");
    let s_col = svd.S().column_vector(); // ColRef<f64>, length k
    let u_mat = svd.U();                 // MatRef<f64>, shape (m, k)
    let v_mat = svd.V();                 // MatRef<f64>, shape (n, k)

    println!("\nSingular values S ({k}):");
    let s_strs: Vec<String> = (0..k).map(|i| format!("{:12.6}", *s_col.get(i))).collect();
    println!("  {}", s_strs.join(""));

    println!("\nVh ({k}x{n})  [V transposed, rows are right singular vectors]:");
    for i in 0..k {
        let row: Vec<String> = (0..n).map(|j| format!("{:10.6}", *v_mat.get(j, i))).collect();
        println!("  {}", row.join(""));
    }

    // Reconstruction: A ≈ U * diag(S) * Vh
    let mut frob_sq = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let reconstructed: f64 = (0..k)
                .map(|l| *u_mat.get(i, l) * *s_col.get(l) * *v_mat.get(j, l))
                .sum();
            let diff = *mat.get(i, j) - reconstructed;
            frob_sq += diff * diff;
        }
    }
    println!("\nReconstruction error ||A - U·S·Vh||_F = {:.2e}", frob_sq.sqrt());
}
