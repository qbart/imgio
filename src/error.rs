use core::fmt;

/// Errors returned by imgio operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Input data is malformed or truncated.
    InvalidData,
    /// An argument (size, format, option) is out of range or inconsistent.
    InvalidArgument,
    /// The output buffer is too small for the result.
    BufferTooSmall { needed: usize, got: usize },
    /// The format or feature is not supported (yet).
    Unsupported,
}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = core::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidData => f.write_str("invalid data"),
            Error::InvalidArgument => f.write_str("invalid argument"),
            Error::BufferTooSmall { needed, got } => {
                write!(f, "buffer too small: needed {needed} bytes, got {got}")
            }
            Error::Unsupported => f.write_str("unsupported"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let e = Error::BufferTooSmall { needed: 16, got: 8 };
        assert_eq!(e.to_string(), "buffer too small: needed 16 bytes, got 8");
    }
}
