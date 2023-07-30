use nalgebra::{DMatrix, DVector};

/// Linear Kalman filter(Dynamic)
///
/// # Examples
///
/// ```
/// use nalgebra::{DMatrix, DVector};
/// use rs_kalman::models::linear::DKalmanFilter;
///
/// let mut kf = DKalmanFilter::new(
///     DVector::<f64>::from_vec(vec![0.0, 0.0]),
///     DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
///     DMatrix::<f64>::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]),
///     DMatrix::<f64>::from_vec(1, 1, vec![0.0]),
///     DMatrix::<f64>::from_vec(1, 2, vec![1.0, 0.0]),
///     DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
///     DMatrix::<f64>::from_vec(2, 1, vec![0.0, 0.0]),
/// );
///
/// kf.predict(None);
/// kf.update(&DVector::<f64>::from_vec(vec![1.0]));
///
/// assert_eq!(kf.x, DVector::<f64>::from_vec(vec![1.0, 0.0]));
/// ```
///
/// # Type parameters
///
/// * `T`: RealField
///
/// # Fields
///
/// * `x`: filter state estimate
/// * `p`: covariance matrix
/// * `q`: process uncertainty/noise
/// * `r`: measurement uncertainty/noise
/// * `h`: measurement function
/// * `f`: state transition matrix
/// * `b`: control transition matrix
///
/// # References
///
/// https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/kalman_filter.py
///
pub struct DKalmanFilter<T: na::RealField> {
    /// filter state estimate
    pub x: DVector<T>,
    /// covariance matrix
    pub p: DMatrix<T>,
    /// process uncertainty/noise
    pub q: DMatrix<T>,
    /// measurement uncertainty/noise
    pub r: DMatrix<T>,
    /// measurement function
    pub h: DMatrix<T>,
    /// state transition matrix
    pub f: DMatrix<T>,
    /// control transition matrix
    pub b: DMatrix<T>,
}

impl<T: na::RealField> DKalmanFilter<T> {
    /// Constructs a new KalmanFilter object with the given initial values for the state estimate, covariance matrix, process noise covariance matrix, measurement noise covariance matrix, measurement function matrix, state transition matrix, and control transition matrix.
    ///
    /// # Arguments
    ///
    /// * `x` - The initial state estimate vector.
    /// * `p` - The initial covariance matrix.
    /// * `q` - The process noise covariance matrix.
    /// * `r` - The measurement noise covariance matrix.
    /// * `h` - The measurement function matrix.
    /// * `f` - The state transition matrix.
    /// * `b` - The control transition matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{DMatrix, DVector};
    /// use rs_kalman::models::linear::DKalmanFilter;
    ///
    /// let x = DVector::<f64>::from_vec(vec![0.0, 0.0]);
    /// let p = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let q = DMatrix::<f64>::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
    /// let r = DMatrix::<f64>::from_vec(1, 1, vec![0.0]);
    /// let h = DMatrix::<f64>::from_vec(1, 2, vec![1.0, 0.0]);
    /// let f = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let b = DMatrix::<f64>::from_vec(2, 1, vec![0.0, 0.0]);
    ///
    /// let mut kf = DKalmanFilter::new(x, p, q, r, h, f, b);
    /// ```
    pub fn new(
        x: DVector<T>,
        p: DMatrix<T>,
        q: DMatrix<T>,
        r: DMatrix<T>,
        h: DMatrix<T>,
        f: DMatrix<T>,
        b: DMatrix<T>,
    ) -> Self {
        Self {
            x,
            p,
            q,
            r,
            h,
            f,
            b,
        }
    }

    /// Predicts the next state estimate and covariance matrix based on the current state estimate and covariance matrix, the process noise covariance matrix, and the state transition matrix.
    ///
    /// # Arguments
    ///
    /// * `u` - An optional control input vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{DMatrix, DVector};
    /// use rs_kalman::models::linear::DKalmanFilter;
    ///
    /// let x = DVector::<f64>::from_vec(vec![0.0, 0.0]);
    /// let p = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let q = DMatrix::<f64>::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
    /// let r = DMatrix::<f64>::from_vec(1, 1, vec![0.0]);
    /// let h = DMatrix::<f64>::from_vec(1, 2, vec![1.0, 0.0]);
    /// let f = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let b = DMatrix::<f64>::from_vec(2, 1, vec![0.0, 0.0]);
    ///
    /// let mut kf = DKalmanFilter::new(x, p, q, r, h, f, b);
    ///
    /// kf.predict(None);
    /// ```
    pub fn predict(&mut self, u: Option<&DVector<T>>) {
        self.x = &self.f * &self.x
            + match u {
                Some(u) => &self.b * u,
                None => DVector::zeros(self.x.len()),
            };
        self.p = &self.f * &self.p * &self.f.transpose() + &self.q;
    }

    /// Updates the state estimate and covariance matrix based on the current state estimate and covariance matrix, the measurement noise covariance matrix, the measurement function matrix, and the measurement vector.
    ///
    /// # Arguments
    ///
    /// * `z` - The measurement vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::{DMatrix, DVector};
    /// use rs_kalman::models::linear::DKalmanFilter;
    ///
    /// let x = DVector::<f64>::from_vec(vec![0.0, 0.0]);
    /// let p = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let q = DMatrix::<f64>::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
    /// let r = DMatrix::<f64>::from_vec(1, 1, vec![0.0]);
    /// let h = DMatrix::<f64>::from_vec(1, 2, vec![1.0, 0.0]);
    /// let f = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    /// let b = DMatrix::<f64>::from_vec(2, 1, vec![0.0, 0.0]);
    ///
    /// let mut kf = DKalmanFilter::new(x, p, q, r, h, f, b);
    ///
    /// kf.update(&DVector::<f64>::from_vec(vec![1.0]));
    /// ```
    pub fn update(&mut self, z: &DVector<T>) {
        let y = z - &self.h * &self.x;
        let s = &self.h * &self.p * &self.h.transpose() + &self.r;
        let k = &self.p * &self.h.transpose() * s.try_inverse().unwrap();
        self.x = &self.x + &k * y;
        self.p = (&DMatrix::<T>::identity(self.p.nrows(), self.p.nrows()) - &k * &self.h) * &self.p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::*;
    use rand::Rng;

    #[test]
    fn test_plot_demo_image() {
        // Generate noisy linear function
        let mut rng = rand::thread_rng();
        let mut y = Vec::new();
        let mut x = Vec::new();
        for i in 0..100 {
            let noise = rng.gen_range(-30.0..30.0);
            let val = 2.0 * i as f64 + noise;
            y.push(val);
            x.push(i as f64);
        }

        // Initialize Kalman filter
        let x_0 = DVector::<f64>::from_vec(vec![0.0, 0.0]);
        let p = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let q = DMatrix::<f64>::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
        let r = DMatrix::<f64>::from_vec(1, 1, vec![1.0]);
        let h = DMatrix::<f64>::from_vec(1, 2, vec![1.0, 0.0]);
        let f = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 0.0, 1.0, 1.0]); // XXX: Different with Static Matrix::new
        let b = DMatrix::<f64>::from_vec(2, 1, vec![0.0, 0.0]);
        let mut kf = DKalmanFilter::new(x_0, p, q, r, h, f, b);

        // Predict and update using Kalman filter
        let mut filtered = Vec::new();
        for i in 0..100 {
            kf.predict(None);
            kf.update(&DVector::<f64>::from_vec(vec![y[i]]));
            filtered.push(kf.x[0].clone());
        }

        // Plot results
        let root = BitMapBackend::new("demo/kalman-filter-linear(dynamic).jpg", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Kalman Filtered Linear(Dynamic)", ("sans-serif", 30))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f64..100f64, -10f64..210f64)
            .unwrap();

        chart.configure_mesh().draw().unwrap();
        chart
            .draw_series(LineSeries::new(
                x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)),
                &RED,
            ))
            .unwrap()
            .label("Noisy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .draw_series(LineSeries::new(
                x.iter().zip(filtered.iter()).map(|(x, y)| (*x, *y)),
                &BLUE,
            ))
            .unwrap()
            .label("Filtered")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();
    }
}
