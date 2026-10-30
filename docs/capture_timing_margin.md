# Capture warning: response starts before its timing reference

The warning `3817 against 3826 samples` compares two different estimates:

* `3817`: the clean loopback's first detected response, minus the NAM
  calibrator's one-sample safety factor;
* `3826`: the label derived from the sub-sample impulse peak, minus the
  capture app's fixed four-sample margin.

The stock margin was based on measurements of one interface. Converter
filters and other bandlimited I/O paths can spread a response further ahead
of its peak. Correctly aligned hardware can therefore trigger this warning.
The warning alone does not establish a Windows driver or routing fault.

The regression test reproduces these exact values using an otherwise stable,
clean route with a 12-sample gap between its first response and its peak.
This demonstrates the software assumption; identifying the response of a
particular physical interface still requires its raw loopback recording.

## Correction

For projects using the current capture timebase, a capture with reliably
detected, agreeing loopback blips can increase the margin automatically.
It is increased by a whole number of samples. Every previous capture's delay
label is decreased by that same number, and the project and `data.json` are
saved together by the capture path. Existing WAV samples and fractional
shifts are unchanged. Consequently all captures retain the same relative
alignment, including when the first capture used the stock application.

The measured margin is stored as `alignment_lead_samples` in the project.
It survives reopening and regenerating the capture plan. Clearing all
captures resets it. In the reproduced example the margin becomes 14
samples, and the corrected label is 3816 instead of 3826.

A configured but missing loopback, disagreeing timing blips, and implausibly
large offsets still produce failures or warnings. They are not used to
calibrate the margin. The amp return is not used to learn this margin,
because its phase response can legitimately change with the amp controls.

If existing WAVs are pending recovery, restore them first so their saved
labels can be included in the same adjustment. Projects carrying an explicit
legacy `alignment_reference` keep that earlier timebase and the existing
warning behavior; automatic legacy migration is intentionally separate.

The QA comparison also accounts for the fractional shift already applied to
the written target: a raw onset and a shifted delay label must be compared
in the same sample coordinates.

Tests cover the exact reported numbers, preservation of previously recorded
audio, alignment audits after margin changes, rejection of unreliable blips,
and rollback of delay labels if writing the dataset fails.
