# Parallel Deferred Correction Integrators
This repo contains rust code for Parallel Deferred Correction integrators.

This code was written for my master's thesis, [Parallel Deferred Correction for the Solution of Initial Value
Problems in Astrodynamics](https://www.proquest.com/openview/760114430fcc2b09540fa9fe6392a592/1?pq-origsite=gscholar&cbl=51922&diss=y)

As far as I am aware this is the only existing Rust implementation of these novel integrators and one of only a handful of RIDC implementations that exist.


## What is Parallel Deferred Correction?
Revisionist Integral Deferred Correction (RIDC) is an extension of Spectral Integral Deferred Correction (IDC) integration techniques. As with a normal spectral IDC method, multiple stages of correction are applied to an initial estimated integral. Revisionist Deferred correction however, uses a moving window for correction rather than using a fixed window like Spectral IDC methods.

Parallel Deferred Correction expands on RIDC techniques to take advantage of the fact that each level of correction depends only on the previous level's corrected estimate. We can split each stage of correction out to a separate thread and pass the corrected values between threads.

![image](https://user-images.githubusercontent.com/32889994/188971349-bacfb713-6a45-4552-b085-5ec16ff39155.png)

When the function being integrated is computationally expensive the separation of correction stages into separate threads can lead to significant computational speed gains while producing a highly accurate solution. 

Both fixed and adaptive versions of the RIDC integrator are implemented in this repo.



## Contents of this Repo
- Runge Kutta Integrators
- Lagrange Interpolation (Divided difference)
- Adams integrators 
- RIDC integrators
- Cheby interpolator

## Future Updates
The following are some updates I hope to eventually get to:
- Use Automatic differentiation for RIDC integrator
- Update the inter-process queues to use ring buffers for improved efficiency 
- Server-based, configurable implementation

