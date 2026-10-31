# ASIO capture timing (capture app 1.3.0)

ASIO audio stays open between route tests and captures on one dedicated audio
thread. This avoids repeatedly restarting the driver and calibrating against its
startup latency. Changing the device, sample rate, channels or native buffer opens
a new stream and starts calibration again. Closing the project/app, refreshing
devices or pressing **Release audio** closes the stream on its owning thread.

In **Audio I/O**, **ASIO buffer size** requests the native buffer size, rather than
a separate software processing block. The actual callback size is checked before
playing audio and shown in the route-test result. An unsupported size is refused.
The separate **Stream latency** control is hidden for ASIO. **Auto** uses the
reported device default; refresh devices after changing a driver's control panel,
or select an explicit supported size. Other audio APIs retain their existing
buffer/latency controls.

With a clean loopback enabled, the first operation sends a low-level calibration
signal only through that loopback, leaving the pedal output silent. It observes
startup for at least 20 seconds and requires repeated agreeing timing measurements;
it fails if it cannot establish a reliable reference within the calibration limit.
The observation floor is not a promise that every device settles in 20 seconds:
each recording is also checked against that reference throughout its measurable
audio. Subsequent captures reuse the open stream and verify the reference again.

If the loopback changes by more than 0.25 samples, or stops providing a reliable
reference, the capture is refused before its WAV and dataset entry are saved. A
retry recalibrates the still-open stream. Cancellation silences playback, and a
driver dropout is still an error. ASIO captures without a clean loopback cannot
verify timing stability and report that limitation.

## Why an error-free ASIO stream is not sufficient

On the tested FM9 / ASIO driver 6.16.0.0, the round trip changed by 7–8 samples
after startup. This reproduced through both PortAudio and a separate native ASIO
host, including a five-minute recording at buffer 2048. Callback counters and
reported ASIO latencies stayed continuous and no reset/resync notification arrived.
The app therefore verifies the recorded clean loopback, not only driver flags.

The spectral timing comparison uses coherent, sufficiently excited frequencies
from the clean route. Converter filtering, fractional delay and a change of signal
level do not by themselves invalidate it. The pedal return is not used to infer
this drift: its frequency-dependent phase is part of what the model must learn.

Existing captures are not rewritten or realigned by this update. Recordings with
startup drift need a separate raw-data recovery or a new capture; a uniform
7-sample adjustment is not correct for a file whose latency changes mid-recording.
**Check captures** also compares the program in the preserved raw loopback with
its initial reference, so agreeing startup blips no longer hide later drift.
The check identifies suspect files without modifying them.
