use na::{SMatrix, SVector};

/// Linear Kalman filter(Static)
///
/// # Examples
///
/// ```
/// use nalgebra as na;
/// use rs_kalman::models::linear::SKalmanFilter;
///
/// let mut kf = SKalmanFilter::new(
///     na::SVector::<f64, 2>::new(0.0, 0.0),
///     na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0),
///     na::SMatrix::<f64, 2, 2>::new(0.0, 0.0, 0.0, 0.0),
///     na::SMatrix::<f64, 1, 1>::new(0.0),
///     na::SMatrix::<f64, 1, 2>::new(1.0, 0.0),
///     na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0),
///     na::SMatrix::<f64, 2, 1>::new(0.0, 0.0),
/// );
///
/// kf.predict(None);
/// kf.update(&na::SVector::<f64, 1>::new(1.0));
///
/// assert_eq!(kf.x, na::SVector::<f64, 2>::new(1.0, 0.0));
/// ```
///
/// # Type parameters
///
/// * `T`: RealField
/// * `X`(Const): Dimension of state vector
/// * `Z`(Const): Dimension of measurement vector
/// * `U`(Const): Dimension of control vector
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
pub struct SKalmanFilter<T: na::RealField, const X: usize, const Z: usize, const U: usize> {
    /// filter state estimate
    pub x: SVector<T, X>,
    /// covariance matrix
    pub p: SMatrix<T, X, X>,
    /// process uncertainty/noise
    pub q: SMatrix<T, X, X>,
    /// measurement uncertainty/noise
    pub r: SMatrix<T, Z, Z>,
    /// measurement function
    pub h: SMatrix<T, Z, X>,
    /// state transition matrix
    pub f: SMatrix<T, X, X>,
    /// control transition matrix
    pub b: SMatrix<T, X, U>,
}

impl<T: na::RealField, const X: usize, const Z: usize, const U: usize> SKalmanFilter<T, X, Z, U> {
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
    /// use nalgebra as na;
    /// use rs_kalman::models::linear::SKalmanFilter;
    ///
    /// let x = na::SVector::<f64, 2>::new(0.0, 0.0);
    /// let p = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
    /// let q = na::SMatrix::<f64, 2, 2>::new(0.0, 0.0, 0.0, 0.0);
    /// let r = na::SMatrix::<f64, 1, 1>::new(0.0);
    /// let h = na::SMatrix::<f64, 1, 2>::new(1.0, 0.0);
    /// let f = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
    /// let b = na::SMatrix::<f64, 2, 1>::new(0.0, 0.0);
    ///
    /// let mut kf = SKalmanFilter::new(x, p, q, r, h, f, b);
    /// ```
    pub fn new(
        x: SVector<T, X>,
        p: SMatrix<T, X, X>,
        q: SMatrix<T, X, X>,
        r: SMatrix<T, Z, Z>,
        h: SMatrix<T, Z, X>,
        f: SMatrix<T, X, X>,
        b: SMatrix<T, X, U>,
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
    /// use nalgebra as na;
    /// use rs_kalman::models::linear::SKalmanFilter;
    ///
    /// let x = na::SVector::<f64, 2>::new(0.0, 0.0);
    /// let p = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
    /// let q = na::SMatrix::<f64, 2, 2>::new(0.0, 0.0, 0.0, 0.0);
    /// let r = na::SMatrix::<f64, 1, 1>::new(0.0);
    /// let h = na::SMatrix::<f64, 1, 2>::new(1.0, 0.0);
    /// let f = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
    /// let b = na::SMatrix::<f64, 2, 1>::new(0.0, 0.0);
    ///
    /// let mut kf = SKalmanFilter::new(x, p, q, r, h, f, b);
    ///
    /// kf.predict(None);
    /// ```
    pub fn predict(&mut self, u: Option<&SVector<T, U>>) {
        self.x = &self.f * &self.x
            + match u {
                Some(u) => &self.b * u,
                None => SVector::zeros(),
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
    /// use nalgebra as na;
    /// use rs_kalman::models::linear::SKalmanFilter;
    ///
    /// let x = na::SVector::<f64, 2>::new(0.0, 0.0);
    /// let p = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
    /// let q = na::SMatrix::<f64, 2, 2>::new(0.0, 0.0, 0.0, 0.0);
    /// let r = na::SMatrix::<f64, 1, 1>::new(0.0);
    /// let h = na::SMatrix::<f64, 1, 2>::new(1.0, 0.0);
    /// let f = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
    /// let b = na::SMatrix::<f64, 2, 1>::new(0.0, 0.0);
    ///
    /// let mut kf = SKalmanFilter::new(x, p, q, r, h, f, b);
    ///
    /// kf.update(&na::SVector::<f64, 1>::new(1.0));
    /// ```
    pub fn update(&mut self, z: &SVector<T, Z>) {
        let y = z - &self.h * &self.x;
        let s = &self.h * &self.p * &self.h.transpose() + &self.r;
        let k = &self.p * &self.h.transpose() * s.try_inverse().unwrap();
        self.x = &self.x + &k * y;
        self.p = (&SMatrix::identity() - &k * &self.h) * &self.p;
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
        let x_0 = SVector::<f64, 2>::new(0.0, 0.0);
        let p = SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, 1.0);
        let q = SMatrix::<f64, 2, 2>::new(0.0, 0.0, 0.0, 0.0);
        let r = SMatrix::<f64, 1, 1>::new(1.0);
        let h = SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let f = SMatrix::<f64, 2, 2>::new(1.0, 1.0, 0.0, 1.0);
        let b = SMatrix::<f64, 2, 1>::new(0.0, 0.0);

        let mut kf = SKalmanFilter::new(x_0, p, q, r, h, f, b);

        // Predict and update using Kalman filter
        let mut filtered = Vec::new();
        for i in 0..100 {
            kf.predict(None);
            kf.update(&SVector::<f64, 1>::new(y[i]));
            filtered.push(kf.x[0].clone());
        }

        // Plot results
        let root =
            BitMapBackend::new("demo/kalman-filter-linear.jpg", (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Kalman Filtered Linear(Static)", ("sans-serif", 30))
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
