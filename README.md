# Neural ODEs: Toy Potentials and Triatomic Harmonic Molecule
This is a repository for the first two problems of my Master's Thesis called Coarse-Graining of Molecular Dynamics Using Neural Ordinary Differential Equations. The project has two parts. Firstly, three different toy potentials are learned with neural ODEs. Then a benchmark is done by performing Hamiltonian Monte Carlo sampling of those potentials, where the dynamics proposal step is evolved with neural ODEs. Secondly, a diatomic harmonic molecule potential is learned and then extended to a triatomic molecule, investigating the feasibility of extending pair-wise interactions.

This project was done as a collaboration between MIT's Rafael Gómez-Bombarelli and Imperial's Stefano Angiolleti-Uberti research groups.

### Toy Potential Results

![alt text](https://github.com/jakublala/md-neural-ode/blob/main/figures/10d_gaussian_hmc.png?raw=true)
