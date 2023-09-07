# LRC-CD: *L*onge-*R*ange *C*orrelation energy model with *C*harges and *D*ipoles.
Copyright (C) 2020-2023 Yingxing Cheng

The ACKS2 model is an innovative PFF model developed by Verstraelen et al. 
This model is entirely derived from KS-DFT, with only a few reasonable approximations applied. 
As a result, it performs admirably, maintaining a high level of accuracy in comparison to DFT calculations. 
The ACKS2-omega model is an extension of the ACKS2 model that accounts for frequency-dependence, allowing it to capture the dynamic responses of molecules to external perturbations, such as fields produced by other molecules. 
Consequently, the ACKS2-omega model can be employed to calculate dispersion energy or long-range correlation energy using ACFD theory.

The LRC-CD model offers a framework to compute both inter-molecular and intra-molecular long-distance correlation energy based on the ACKS2-omega model. 
At its core, LRC-CD leverages DFT density to determine long-range correlation energy, utilizing ACFD theory combined with RPA methods. 
These RPA methods are typically used in conventional TDDFT calculations. 
The LRC-CD model integrates the ACKS2 model to achieve molecular linear-response properties. 
This method has been thoroughly validated against more resource-intensive computational chemistry techniques. 
Thus, it presents an alternative approach for evaluating linear response properties of interacting systems.
Typically, such evaluations involve using a Dyson-like equation that originates from a non-interacting system (e.g., the KS system) and the response function within the TDDFT regime.

## Installation

### From Source

```bash
git clone https://github.com/YourUsername/lrccd.git
cd lrccd
pip install .
```

### Testing
```bash
pytest tests
```
