///
/// bliss-learner — Metric learning for bliss-mixer
///
/// Learns a Mahalanobis distance matrix from user-provided "odd-one-out"
/// training triplets (JSON file) and bliss features (bliss.db).
///
/// Based on bliss-metric-learning by Polochon-street
/// (https://github.com/Polochon-street/bliss-metric-learning)
///
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array1, Array2, Axis, Order};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rusqlite::Connection;
use serde_json::json;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIMENSIONS: usize = 23;

const FEATURE_COLUMNS: [&str; DIMENSIONS] = [
    "Tempo", "Zcr",
    "MeanSpectralCentroid", "StdDevSpectralCentroid",
    "MeanSpectralRolloff", "StdDevSpectralRolloff",
    "MeanSpectralFlatness", "StdDevSpectralFlatness",
    "MeanLoudness", "StdDevLoudness",
    "Chroma1", "Chroma2", "Chroma3", "Chroma4", "Chroma5",
    "Chroma6", "Chroma7", "Chroma8", "Chroma9", "Chroma10",
    "Chroma11", "Chroma12", "Chroma13",
];

const SIGMA: f64 = 2.0;

const LAMBDAS: [f64; 10] = [0.0, 0.001, 0.01, 0.1, 1.0, 50.0, 100.0, 500.0, 1000.0, 5000.0];

const MIN_NOTIF_TIME: u64 = 2;

// ---------------------------------------------------------------------------
// LMS Notification (same pattern as bliss-analyser)
// ---------------------------------------------------------------------------

struct NotifInfo {
    enabled: bool,
    address: String,
    last_send: u64,
    start_time: u64,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("time").as_secs()
}

fn send_notif_msg(notifs: &mut NotifInfo, text: &str) {
    let js = json!({
        "id": "1",
        "method": "slim.request",
        "params": ["", ["blissmixer", "survey", "act:update", format!("msg:{}", text)]]
    });
    log::info!("Sending notif to LMS: {}", text);
    let _ = ureq::post(&notifs.address).send_string(&js.to_string());
}

fn send_notif(notifs: &mut NotifInfo, text: &str) {
    if notifs.enabled {
        let now = now_secs();
        if now >= notifs.last_send + MIN_NOTIF_TIME {
            let dur = now - notifs.start_time;
            let msg = format!("[{:02}:{:02}:{:02}] {}", dur / 3600, (dur / 60) % 60, dur % 60, text);
            send_notif_msg(notifs, &msg);
            notifs.last_send = now;
        }
    }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

type Triplet = [Array1<f64>; 3];

fn load_triplets(db_path: &str, triplets_path: &str) -> Vec<Triplet> {
    let json_str = std::fs::read_to_string(triplets_path)
        .expect("Cannot read triplets file");
    let raw: Vec<Vec<String>> = serde_json::from_str(&json_str)
        .expect("Cannot parse triplets JSON");

    let conn = Connection::open(db_path).expect("Cannot open database");
    let cols = FEATURE_COLUMNS.join(", ");
    let feature_sql = format!("SELECT {} FROM TracksV2 WHERE File = ?1", cols);

    let mut triplets = Vec::new();
    let mut skipped = 0u32;

    for entry in &raw {
        if entry.len() != 3 { skipped += 1; continue; }
        let mut features: Vec<Array1<f64>> = Vec::with_capacity(3);
        let mut ok = true;
        for file in entry {
            match conn.query_row(&feature_sql, [file], |row| {
                let mut vals = vec![0.0f64; DIMENSIONS];
                for i in 0..DIMENSIONS {
                    vals[i] = row.get(i)?;
                }
                Ok(Array1::from(vals))
            }) {
                Ok(arr) => features.push(arr),
                Err(_) => { skipped += 1; ok = false; break; }
            }
        }
        if ok && features.len() == 3 {
            triplets.push([
                features.remove(0),
                features.remove(0),
                features.remove(0),
            ]);
        }
    }

    if skipped > 0 {
        log::warn!("Skipped {} triplets with missing tracks", skipped);
    }

    triplets
}

// ---------------------------------------------------------------------------
// Core math — direct port from bliss-metric-learning
// ---------------------------------------------------------------------------

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Mahalanobis distance: sqrt((x1-x2)^T L L^T (x1-x2))
fn d(l_flat: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
    let n = x1.len();
    let l_mat = l_flat.view().into_shape_with_order(((n, n), Order::RowMajor)).expect("reshape L");
    let diff = x1 - x2;
    let tmp = diff.dot(&l_mat.dot(&l_mat.t()));
    let sqrd = tmp.dot(&diff);
    sqrd.sqrt()
}

fn grad_d_squared(l_flat: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>) -> Array1<f64> {
    let n = x1.len();
    let l_mat = l_flat.view().into_shape_with_order(((n, n), Order::RowMajor)).expect("reshape L");
    let diff = x1 - x2;
    // grad = 2 * outer(diff, diff) . L  →  flattened
    let outer = outer_product(&diff, &diff);
    let grad = outer.dot(&l_mat) * 2.0;
    grad.into_shape_with_order((n * n, Order::RowMajor)).expect("flatten grad").to_owned()
}

fn grad_d(l_flat: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>) -> Array1<f64> {
    let dist = d(l_flat, x1, x2);
    &grad_d_squared(l_flat, x1, x2) / (2.0 * dist)
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let a_col = a.view().insert_axis(Axis(1));         // (n, 1)
    let b_row = b.view().insert_axis(Axis(0));         // (1, m)
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a_col[[i, 0]] * b_row[[0, j]];
        }
    }
    result
}

/// x3 is the odd one out
fn delta(l: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>, x3: &Array1<f64>,
         sigma: f64, second_batch: bool) -> f64 {
    if second_batch {
        (d(l, x1, x3) - d(l, x1, x2)) / sigma
    } else {
        (d(l, x2, x3) - d(l, x1, x2)) / sigma
    }
}

fn grad_delta(l: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>, x3: &Array1<f64>,
              sigma: f64, second_batch: bool) -> Array1<f64> {
    if second_batch {
        (&grad_d(l, x1, x3) - &grad_d(l, x1, x2)) / sigma
    } else {
        (&grad_d(l, x2, x3) - &grad_d(l, x1, x2)) / sigma
    }
}

fn p_triplet(l: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>, x3: &Array1<f64>,
             sigma: f64, second_batch: bool) -> f64 {
    norm_cdf(delta(l, x1, x2, x3, sigma, second_batch))
}

fn grad_p(l: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>, x3: &Array1<f64>,
          sigma: f64, second_batch: bool) -> Array1<f64> {
    let pdf = norm_pdf(delta(l, x1, x2, x3, sigma, second_batch));
    &grad_delta(l, x1, x2, x3, sigma, second_batch) * pdf
}

fn log_p(l: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>, x3: &Array1<f64>,
         sigma: f64, second_batch: bool) -> f64 {
    p_triplet(l, x1, x2, x3, sigma, second_batch).ln()
}

fn grad_log_p(l: &Array1<f64>, x1: &Array1<f64>, x2: &Array1<f64>, x3: &Array1<f64>,
              sigma: f64, second_batch: bool) -> Array1<f64> {
    let pv = p_triplet(l, x1, x2, x3, sigma, second_batch);
    &grad_p(l, x1, x2, x3, sigma, second_batch) / pv
}

/// Negative log-likelihood + L2 regularisation
fn opti_fun(l: &Array1<f64>, x: &[Triplet], sigma: f64, lambda: f64) -> f64 {
    let mut batch1 = 0.0;
    let mut batch2 = 0.0;
    for t in x {
        batch1 -= log_p(l, &t[0], &t[1], &t[2], sigma, false);
        batch2 -= log_p(l, &t[0], &t[1], &t[2], sigma, true);
    }
    batch1 + batch2 + lambda * l.mapv(|v| v * v).sum()
}

fn grad_opti_fun(l: &Array1<f64>, x: &[Triplet], sigma: f64, lambda: f64) -> Array1<f64> {
    let n = l.len();
    let mut batch1 = Array1::zeros(n);
    let mut batch2 = Array1::zeros(n);
    for t in x {
        batch1 -= &grad_log_p(l, &t[0], &t[1], &t[2], sigma, false);
        batch2 -= &grad_log_p(l, &t[0], &t[1], &t[2], sigma, true);
    }
    &batch1 + &batch2 + &(l * (2.0 * lambda))
}

fn percentage_preserved_distances(l_mat: &Array2<f64>, x: &[Triplet]) -> f64 {
    let l_flat = l_mat.clone().into_shape_with_order((DIMENSIONS * DIMENSIONS, Order::RowMajor))
        .expect("flatten L").to_owned();
    let mut count = 0usize;
    for t in x {
        let d1 = d(&l_flat, &t[0], &t[1]); // short distance (similar pair)
        let d2 = d(&l_flat, &t[1], &t[2]); // long distance
        let d3 = d(&l_flat, &t[0], &t[2]); // long distance
        if d1 < d2 && d1 < d3 {
            count += 1;
        }
    }
    count as f64 / x.len() as f64
}

// ---------------------------------------------------------------------------
// L-BFGS-B optimizer (numpy-only fallback ported to Rust)
// ---------------------------------------------------------------------------

fn lbfgs_minimize<F, G>(
    fun: F,
    x0: &Array1<f64>,
    jac: G,
    m: usize,
    maxiter: usize,
    gtol: f64,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x = x0.clone();
    let mut s_hist: Vec<Array1<f64>> = Vec::new();
    let mut y_hist: Vec<Array1<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let mut f = fun(&x);
    let mut g = jac(&x);

    for _ in 0..maxiter {
        // Convergence check
        if g.mapv(f64::abs).fold(0.0f64, |a, &b| a.max(b)) < gtol {
            break;
        }

        // Two-loop recursion for search direction
        let mut q = g.clone();
        let mut alphas: Vec<f64> = Vec::new();
        for i in (0..s_hist.len()).rev() {
            let a = rho_hist[i] * s_hist[i].dot(&q);
            alphas.push(a);
            q = &q - &(&y_hist[i] * a);
        }
        alphas.reverse();

        let r = if !s_hist.is_empty() {
            let sk = &s_hist[s_hist.len() - 1];
            let yk = &y_hist[y_hist.len() - 1];
            &q * (sk.dot(yk) / yk.dot(yk))
        } else {
            q.clone()
        };

        let mut r = r;
        for i in 0..s_hist.len() {
            let b = rho_hist[i] * y_hist[i].dot(&r);
            r = &r + &(&s_hist[i] * (alphas[i] - b));
        }

        let direction = -&r;

        // Armijo backtracking line search
        let dg = g.dot(&direction);
        let mut alpha = 1.0;
        let mut x_new;
        let mut f_new;
        let mut found = false;
        for _ in 0..20 {
            x_new = &x + &(&direction * alpha);
            f_new = fun(&x_new);
            if f_new <= f + 1e-4 * alpha * dg {
                found = true;
                break;
            }
            alpha *= 0.5;
        }
        x_new = &x + &(&direction * alpha);
        f_new = fun(&x_new);

        let g_new = jac(&x_new);
        let s_k = &x_new - &x;
        let y_k = &g_new - &g;
        let sy = s_k.dot(&y_k);

        if sy > 1e-10 {
            if s_hist.len() >= m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_k);
            y_hist.push(y_k);
            rho_hist.push(1.0 / sy);
        }

        x = x_new;
        f = f_new;
        g = g_new;

        let _ = found; // suppress unused warning
    }

    x
}

fn optimize(l0: &Array1<f64>, x: &[Triplet], sigma: f64, lambda: f64) -> Array2<f64> {
    let x_opt = lbfgs_minimize(
        |l| opti_fun(l, x, sigma, lambda),
        l0,
        |l| grad_opti_fun(l, x, sigma, lambda),
        10,   // history size
        200,  // max iterations
        1e-5, // gradient tolerance
    );
    x_opt.into_shape_with_order(((DIMENSIONS, DIMENSIONS), Order::RowMajor)).expect("reshape result").to_owned()
}

// ---------------------------------------------------------------------------
// Cross-validation helpers
// ---------------------------------------------------------------------------

fn split_train_test(x: &[Triplet], test_fraction: f64, seed: u64) -> (Vec<Triplet>, Vec<Triplet>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.shuffle(&mut rng);
    let split = (x.len() as f64 * (1.0 - test_fraction)) as usize;
    let train: Vec<Triplet> = indices[..split].iter().map(|&i| x[i].clone()).collect();
    let test: Vec<Triplet> = indices[split..].iter().map(|&i| x[i].clone()).collect();
    (train, test)
}

fn kfold_indices(n: usize, n_splits: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let fold_size = n / n_splits;
    let mut folds = Vec::new();
    for i in 0..n_splits {
        let start = i * fold_size;
        let end = if i < n_splits - 1 { start + fold_size } else { n };
        let test_idx: Vec<usize> = indices[start..end].to_vec();
        let mut train_idx: Vec<usize> = indices[..start].to_vec();
        train_idx.extend_from_slice(&indices[end..]);
        folds.push((train_idx, test_idx));
    }
    folds
}

fn select_by_indices(x: &[Triplet], indices: &[usize]) -> Vec<Triplet> {
    indices.iter().map(|&i| x[i].clone()).collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let mut db_path = String::new();
    let mut triplets_path = String::new();
    let mut output_path = String::new();
    let mut lms_host = String::from("127.0.0.1");
    let mut lms_json_port: u16 = 9000;
    let mut send_notifs = false;
    let mut logging = String::from("info");

    {
        let mut ap = argparse::ArgumentParser::new();
        ap.set_description("Metric learning for bliss-mixer");
        ap.refer(&mut db_path).add_option(&["-d", "--db"], argparse::Store, "Path to bliss.db");
        ap.refer(&mut triplets_path).add_option(&["-t", "--triplets"], argparse::Store, "Path to training_triplets.json");
        ap.refer(&mut output_path).add_option(&["-o", "--output"], argparse::Store, "Output path for learned_matrix.json");
        ap.refer(&mut lms_host).add_option(&["-L", "--lms"], argparse::Store, "LMS hostname (default: 127.0.0.1)");
        ap.refer(&mut lms_json_port).add_option(&["-J", "--json"], argparse::Store, "LMS JSON-RPC port (default: 9000)");
        ap.refer(&mut send_notifs).add_option(&["-N", "--notifs"], argparse::StoreTrue, "Send progress notifications to LMS");
        ap.refer(&mut logging).add_option(&["-l", "--logging"], argparse::Store, "Log level (default: info)");
        ap.parse_args_or_exit();
    }

    // Set up logging
    let log_level = match logging.to_lowercase().as_str() {
        "error" => log::LevelFilter::Error,
        "warn" => log::LevelFilter::Warn,
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        _ => log::LevelFilter::Info,
    };
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp(None)
        .init();

    if db_path.is_empty() {
        log::error!("--db is required");
        std::process::exit(1);
    }
    if triplets_path.is_empty() {
        log::error!("--triplets is required");
        std::process::exit(1);
    }
    if output_path.is_empty() {
        log::error!("--output is required");
        std::process::exit(1);
    }

    let mut notifs = NotifInfo {
        enabled: send_notifs,
        address: format!("http://{}:{}/jsonrpc.js", lms_host, lms_json_port),
        last_send: 0,
        start_time: now_secs(),
    };

    // --- Load triplets ---
    send_notif(&mut notifs, "Loading triplets...");
    let triplets = load_triplets(&db_path, &triplets_path);
    let n_triplets = triplets.len();

    if n_triplets < 5 {
        log::error!("Too few valid triplets ({}). Need at least 5.", n_triplets);
        send_notif(&mut notifs, &format!("Error: too few triplets ({})", n_triplets));
        if send_notifs {
            send_notif_msg(&mut notifs, "FINISHED");
        }
        std::process::exit(1);
    }

    log::info!("Loaded {} triplets, {} features", n_triplets, DIMENSIONS);
    send_notif(&mut notifs, &format!("Loaded {} triplets, {} features", n_triplets, DIMENSIONS));

    // --- Identity init ---
    let l0 = Array2::<f64>::eye(DIMENSIONS).into_shape_with_order((DIMENSIONS * DIMENSIONS, Order::RowMajor))
        .expect("flatten eye").to_owned();

    // --- Euclidean baseline ---
    let l_eye = Array2::<f64>::eye(DIMENSIONS);
    let euclidean_acc = percentage_preserved_distances(&l_eye, &triplets);
    log::info!("Euclidean baseline accuracy: {:.1}%", euclidean_acc * 100.0);
    send_notif(&mut notifs, &format!("Euclidean baseline: {:.1}%", euclidean_acc * 100.0));

    // --- 80/20 train/test split ---
    let (design, test) = split_train_test(&triplets, 0.2, 42);
    log::info!("Split: {} design, {} test", design.len(), test.len());

    // --- Cross-validation ---
    let n_splits = std::cmp::min(5, design.len());
    let folds = kfold_indices(design.len(), n_splits, 42);
    let mut accuracies: Vec<Vec<f64>> = vec![Vec::new(); LAMBDAS.len()];

    for (fold_idx, (train_idx, test_idx)) in folds.iter().enumerate() {
        let x_train = select_by_indices(&design, train_idx);
        let x_test = select_by_indices(&design, test_idx);

        log::info!("CV fold {}/{}", fold_idx + 1, n_splits);
        send_notif(&mut notifs, &format!("CV fold {}/{}", fold_idx + 1, n_splits));

        for (i, &lambda) in LAMBDAS.iter().enumerate() {
            send_notif(&mut notifs, &format!(
                "CV fold {}/{}, lambda {}/{}", fold_idx + 1, n_splits, i + 1, LAMBDAS.len()
            ));
            let l = optimize(&l0, &x_train, SIGMA, lambda);
            let accuracy = percentage_preserved_distances(&l, &x_test);
            accuracies[i].push(accuracy);
            log::debug!("  Fold {}, lambda={}: accuracy={:.4}", fold_idx + 1, lambda, accuracy);
        }
    }

    // --- Find best lambda ---
    let mean_accuracies: Vec<f64> = accuracies.iter()
        .map(|a| a.iter().sum::<f64>() / a.len() as f64)
        .collect();

    let best_idx = mean_accuracies.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let best_lambda = LAMBDAS[best_idx];
    let best_cv_accuracy = mean_accuracies[best_idx];

    log::info!("Best lambda={}, CV accuracy={:.1}%", best_lambda, best_cv_accuracy * 100.0);
    send_notif(&mut notifs, &format!(
        "CV done: best lambda={}, accuracy={:.1}%", best_lambda, best_cv_accuracy * 100.0
    ));

    // --- Train on design split, evaluate on test split ---
    let l = optimize(&l0, &design, SIGMA, best_lambda);
    let test_euclidean = percentage_preserved_distances(&l_eye, &test);
    let test_learned = percentage_preserved_distances(&l, &test);
    log::info!(
        "Test set: euclidean={:.1}%, learned={:.1}%",
        test_euclidean * 100.0, test_learned * 100.0
    );

    // --- Final training on ALL data ---
    log::info!("Training final model on all data...");
    send_notif(&mut notifs, "Training final model...");

    let l_total = optimize(&l0, &triplets, SIGMA, best_lambda);
    let m = l_total.dot(&l_total.t());

    // --- Save matrix ---
    let data: Vec<f64> = m.iter().cloned().collect();
    let output = json!({
        "m": {
            "v": 1,
            "dim": [DIMENSIONS, DIMENSIONS],
            "data": data,
        }
    });

    std::fs::write(&output_path, serde_json::to_string(&output).expect("serialize"))
        .expect("write output file");

    let final_accuracy = percentage_preserved_distances(&l_total, &triplets);
    log::info!(
        "Complete: accuracy {:.1}% (was {:.1}% euclidean). Saved to {}",
        final_accuracy * 100.0, euclidean_acc * 100.0, output_path
    );
    send_notif(&mut notifs, &format!(
        "Complete: accuracy {:.1}% (was {:.1}%)", final_accuracy * 100.0, euclidean_acc * 100.0
    ));

    if send_notifs {
        send_notif_msg(&mut notifs, "FINISHED");
    }
}
