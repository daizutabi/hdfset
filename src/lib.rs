use pyo3::prelude::*;

#[pyfunction]
fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[pymodule]
fn hdfset(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
    }
}
