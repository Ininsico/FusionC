package io.fusionc;

/** Fatal front-end / semantic error. The compiler exits non-zero when thrown. */
public class FrontendException extends RuntimeException {

    public FrontendException(String message) {
        super(message);
    }

    public FrontendException(String message, Throwable cause) {
        super(message, cause);
    }
}
