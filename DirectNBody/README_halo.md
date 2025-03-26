
# GetHaloDensityProfile 

### Python script to compute halo density and velocity profiles from Rockstar BGC2 binary outputs.
&nbsp;

## **Execution of the profiling routine**

The script is executed with the following command (with some example values):

    mpiexec -n 2 python -m mpi4py GetHaloDensityProfile.py -i halo00037 -o halo00037 -Nmin 3000 -Nb 15 -rmin 1e-3

With the following options:

* **-n** &emsp; &emsp;&nbsp;&emsp; \[int\] Number of threads (for mpiexec)
* **-i** &emsp; &emsp; &nbsp;&emsp; \[string\] Input file name (without snapshot number)
* **-o** &emsp; &emsp; &nbsp;&emsp;\[string\] Output file name
* **-pf** &emsp; &emsp;&nbsp;&emsp;\[string\] Location of the yaml file file containg the cosmological parameters
* **-Mdef**  &ensp;&nbsp;&emsp; \[string\] Mass definition (default = m200c)
* **-Nmin** &emsp;&emsp;\[int\] Minimum halo particle number (default = 3000)
* **-Mmax** &ensp;&emsp;\[double\] Maximum halo mass in Msol/h (default = 1e20)
* **-rmax** &ensp;&nbsp;&emsp; \[double\] Maximum radius for the binned profile in units of $r_{500}$ (default = 5.0)
* **-rmin** &emsp;&nbsp;&emsp;\[double\] Minimum radius for the binned profile in units of $r_{500}$ (default = 1.0e-3)
* **-Nb** &emsp; &nbsp; &nbsp;&emsp;\[int\] Number of bins (default = Nmin/200)
* **-pt** &emsp; &emsp;&emsp; \[int bool\] Plot some example profiles (default = 0)
* **-ns** &emsp; &nbsp; &nbsp; &emsp; \[string\] Name of the simulation (default = None)
&nbsp;

## **Main program (get_halo_density_profile)** 
The main function of the script is get_halo_density_profile. It can read multiple .bgc2 snapshots in parallel and, based on the parameters given by command line, generates a .hdf5 file containing mass density, number density, velocity and velocity dispersion profiles, both 3D and projected, for every halo. It also computes the halo $r_{500}$ and $r_{200}$. This is done through the [density_profile](#densityprofile-halo-particles-halorcol-r500-haloposcols-halovelcols-particlemass-particleposcols-particlevelcols-z-h-om-ol-massscale-rmin-rmax-nbins) function, called after making the relevant quantities (e.g. halo and particle positions) indipendent from cosmology by multiplying by $a/h$. Functions from IO.py are also used to manage the .bgc2 and .hdf5 files, and write the profiles to file.
&nbsp;

## **Functions definitions**

* ### ***spherical_coordinates*** *(cartesian)*

    Convert 3D vectors from Cartesian coordinates to spherical polar coordinates. It calculates the radial distance, inclination angle, and azimuthal angle for each vector.

    #### **Args:**

    - **cartesian:** *\[ndarray\], shape = (n_vectors, 3)*
        
        Array of 3D vectors in Cartesian coordinates ($x, y, z$).

    #### **Returns:**

    - **spherical:** *\[ndarray\], shape = (n_vectors, 3)*
    
        Array of vectors in spherical coordinates ($r, \theta, \phi$). The columns represent radial distance ($r$), inclination angle ($\theta$), and azimuthal angle ($\phi$).
&nbsp;

* ### ***spherical_velocities*** *(coord_cartesian, vel_cartesian)*

    Convert velocities from Cartesian coordinates to spherical polar coordinates. It first calculates the spherical coordinates for positions and then transforms the Cartesian velocities into spherical coordinates. The result represents velocities in the radial direction, inclination direction, and azimuthal direction.
    

    #### **Args:**

    - **coord_cartesian:** *\[ndarray\], shape = (n_vectors, 3)* 

        Array of 3D vectors in Cartesian coordinates ($x, y, z$).

    - **vel_cartesian:** *\[ndarray\], shape = (n_vectors, 3)* 
        
        Array of 3D velocities in Cartesian coordinates ($v_x, v_y, v_z$).

    #### **Returns:**

    - **vel_spherical:** *\[ndarray\], shape = (n_vectors, 3)*
    
        Array of velocities in spherical coordinates ($v_r, v_\theta, v_\phi$). The columns represent radial velocity ($v_r$), velocity in inclination direction ($v_\theta$), and azimuthal velocity ($v_\phi$).
&nbsp;

 * ### ***get_profile*** *(radius, quantity, profile_type, velocity_dimensions, bin_radius, nbins, rmin, rmax)*

    General profiling function for Rockstar output. **All profiles are in units of $r_{500}$**.

    This function generates profiles for Rockstar output based on the provided parameters, calculating different profiles
    such as density, velocity, or velocity dispersion based on the specified profile type. The function divides the provided radius and quantity into bins, computing various quantities depending on the profile type.
    
    For the *'density'* profile it calculates bin quantities and densities in each volume shell. The i-th volume shell is given by (where $r_i$ is the i-th radius bin edge in log-space):

    $$\Delta V = \frac{4}{3}\pi(r_i^3 - r_{i-1}^3) $$

    The same holds for the *'density2d'* projected profile, but only the particles inside a cylinder of volume $5 \cdot r_{500}$ are considered:

    $$\Delta S =  4\pi (r_i^2 - r_{i-1}^2) \cdot 5$$
    
    For the *'velocity'* and *'velocity disp'* profiles, it computes binned quantities and radii based on the velocities or velocity disperisons provided. The *'velocity+disp'* profile computes both *'velocity'* and *'velocity disp'* at the same time.

    #### **Args:**

    - **radius:** *\[float array-like\]*
    
        1-dimensional array of particle radial positions to be associated to the quantity, in units of $r_{500}$.

    - **quantity:** *\[array-like\]*
    
         If profile_type is *'density'* or *'density2d'*, it should be a 1-dimensional, length len(radius) array of particle masses (to compute mass densities) or particle numbers (to compute number densities). If profile_type is *'velocity'*, *'velocity disp'* or *'velocity+disp'*, it should be a shape (len(radius), dim) array of particle velocities.

    - **profile_type:** *\[str, optional\], {'density', 'density2d', 'velocity', 'velocity disp', 'velocity+disp'}, default = 'density'*
    
        Type of profile to be computed. Not case-sensitive.

    - **velocity_dimensions:** *\[int, optional\], default = 1*
        
        Number of dimensions of the quantity for *'velocity'* or *'velocity disp'* profile.

    - **bin_radius:** *\[array, optional\], default = np.ones(1)* 
    
        Mean radius in each bin, used for *'velocity'*, *'velocity disp'* or *'velocity+disp'* profiles.

    - **nbins:** *\[int, optional\], default = 50* 
        
        Number of bins in the profile

    - **rmin:** *\[float, optional\], default = 1.0e-3*
        
        Minimum radius in units of $r_{500}$.

    - **rmax:** *\[float, optional\], default = 5.0*
        
        Maximum radius in units of $r_{500}$.

    #### **Returns:**

    - *\[tuple\]*: Depending on the profile type, returns different results. 
        1. For *'density'* and *'density2d'* profiles:
            - **mass_weighted_radii:** *\[array\]*
            
                Total radius \* quantity in every bin.

            - **densities_binned:** *\[array\]* 
                
                Calculated average density in each spherical shell.

            - **masses_binned:** *\[array\]*
                
                Total mass in each bin.

            - **mass_cum:** *\[array\]*

                Cumulative sum of the total binned masses.

        2. For *'velocity'*, *'velocity disp'* and *'velocity+disp'* profiles:
            - **bin_centres:** *\[array\]*
                
                Centres of the bin edges in log-scale.

            - **binned_dimensions:** *\[dict\]*: 
                
                Dictionary containing the calculated binned radii and velocities for the dimensions entered.

    #### **Raises:**

    This function uses numpy for calculations and raises specific ValueError or RuntimeError exceptions for unsupported or invalid inputs.

    *ValueError*: If an unsupported profile type is provided or if the inner radius is not smaller than the outer radius.

    *RuntimeError*: If the number of dimensions is insufficient for the *'velocity'* profile.
&nbsp;

* ### ***density_profile*** *(halo, particles, halo_r_col, r500, halo_pos_cols, halo_vel_cols, particle_mass, particle_pos_cols, particle_vel_cols, z, h, Om, Ol, mass_scale, rmin, rmax, nbins)*
    
    For each halo given as input, generates all the 2D-3D profiles.

    This function reads the input halo and the particles belonging to it to derive all the binned radial profiles,
    in three dimensions and for all the projections, using the [get_profile](#get_profile-radius-quantity-profile_type-velocity_dimensions-bin_radius-nbins-rmin-rmax) function. Additionaly, $r_{500}$ and $r_{200}$ are computed using the [calculate_rDelta](#calculaterdelta-r-mass-delta-z-h0-om-ol) function. 

    #### **Args:**

    - **halo:** *\[array-like\], shape = (12, )*
        
        1D-array of floats, containing information about the halos (e.g. positions and velocities).

    - **particles:** *\[array-like\], shape = (Nparticles, 7)*
    
        Multi-dimensional array, containing information about each single particle (e.g. positions and velocities) in the halo. 

    - **halo_r_col:** *\[int, optional\], default = 4* 
        
        Integer indicating the column of halo storing the virial radius. 

    - **dist_scale:** *\[float, optional\], default = 1.0e3*
    
        Physical scale for distances. Default is Mpc (corresponding to 1.0e3).

    - **r500:** *\[float, optional\], default = None*
    
        Value of $r_{500}$ for the halos, used to scale positions in units of $r_{500}$. If None, r_virial is assumed.

    - **halo_pos_cols:** *\[list, optional\], default = []*
        
        List of integers, indexes of columns indicating x, y, z coordinates in the *halo* array. If [], then [6,7,8] is assumed.

    - **halo_vel_cols:** *\[list, optional\], default = []*
    
        List of integers, indexes of columns indicating vx, vy, vz velocities in the *halo* array. If [], then [9,10,11] is assumed.

    - **particle_mass:** *\[float, optional\], default = 1.0*
        
        Value of the mass of the individual particles in Solar Masses. 

    - **particle_pos_cols:** *\[list, optional\], default = []*: 
        
        List of integers, indexes of columns indicating x, y, z coordinates in the *particles* array. If [], then [1,2,3] is assumed.

    - **particle_vel_cols:** *\[list, optional\], default = []*: 
        
        List of integers, indexes of columns indicating vx, vy, vz velocities in the *particles* array. If [], then [4,5,6] is assumed.

    - **z:** *\[float, optional\], default = 0.0*:

        Redshift of the snapshot.

    - **h:** *\[float, optional\], default = 0.67*: 
    
        Value of the hubble paramater H0/100.

    - **Om:** *\[float, optional\], default = 0.31* 
        
        Omega matter at redshift zero.

    - **Ol:** *\[float, optional\], default = 0.689* 

        Omega Lambda at redshfit zero.

    - **rmin:** *\[float, optional\], default = 1e-3* 
        
        Minimum radius for the binning of the profiles, in unit of $r_{500}$.

    - **rmax:** *\[float, optional\], default = 5.0* 
        
        Maximum radius for the binning of the profiles, in unit of $r_{500}$.

    - **nbins:** *\[int, optional\], default = 50*

        Number of bins in the profile.

    #### **Returns:** 

    - *\[tuple\]*: All binned quantities.
        - **rbins:** *\[array-like\], shape = (nbins, 4)* 

            Multi-dimensional array of floats, in units of $r_{500}$, containing the binned 3D radius, and the three projected radii R2Dx,y,z at indexes (0,1,2,3),  respectively.

        - **MassDensity:** *\[array-like\], shape = (nbins, 6)*:  
        
            Multi-dimensional array that contains density, mass of the shell, cumulative mass, and the same for the number of particles, at indexes (0,1,2,3,4,5), respectively.

        - **MassDensity2Dx:** *\[array-like\], shape = (nbins, 6)*

            Multi-dimensional array that contains density, mass of the shell, cumulative mass, projected along the x direction, and the same for the x-projected number of particles, at indexes (0,1,2,3,4,5) respectively.

        - **MassDensity2Dy:** *\[array-like\], shape = (nbins, 6)* 
            
           Same as MassDensity2Dx but for the y projection. 

        - **MassDensity2Dz:** *\[array-like\], shape = (nbins, 6)* 
            
           Same as MassDensity2Dx but for the z projection.

        - **Vel3D:** *\[array-like\], shape = (nbins, 6)*  
        
            Multi-dimensional array that contains all 3D-binned, Cartesian and spherical velocity components, (vx, vy, vz, vr, vt, vph). 

        - **VelDisp3D:** *\[array-like\], shape = (nbins, 6)*
        
            Multi-dimensional array that contains all 3D-binned, Cartesian and spherical velocity dispersion components, ($\sigma_{vx}, \sigma_{vy}, \sigma_{vz}, \sigma_{vr}, \sigma_{vt}, \sigma_{vph}$).   

        - **Vel2Dx:** *\[array-like\], shape = (nbins, 6)* 
            
            Multi-dimensional array that contains all 2D-binned, Cartesian and spherical velocity components, projected along the x direction, (vx(R_x), vy(R_x), vz(R_x), vr(R_x), vt(R_x), vph(R_x)). 

        - **Vel2Dy:** *\[array-like\], shape = (nbins, 6)*
        
            Same as Vel2Dx but for the y projection. 

        - **Vel2Dz:** *\[array-like\], shape = (nbins, 6)*
        
            Same as Vel2Dx but for the z projection. 

        - **VelDisp2Dx** *\[array-like\], shape = (nbins, 6)* 
        
            Multi-dimensional array that contains all 2D-binned velocity dispersion components, projected along x direction, ($\sigma_{vx}, \sigma_{vy}, \sigma_{vz}, \sigma_{vr}, \sigma_{vt}, \sigma_{vph}$).

        - **VelDisp2Dy:**  *\[array-like\], shape = (nbins, 6)*

            Same as VelDisp2Dx but for the y projection. 

        - **VelDisp2Dz:**  *\[array-like\], shape = (nbins, 6)*

            Same as VelDisp2Dx but for the z projection. 

        - **r200:** *\[float\]* 

            Estimated value of $r_{200}$.   

        - **r500:** *\[float\]* 

            Estimated value of $r_{500}$.       
&nbsp;

* ### ***calculate_rDelta*** *(r, mass, Delta, z, H0, Om, Ol)*
    
    Compute the radius enclosing a mass where the density
    is Delta times the critical density of the universe at redshift z, given by:
    $$r_\Delta = \Delta \cdot \frac{H(z)}{2G}$$

    #### **Args:**

    - **r:** *\[float ndarray\]*
        
        Distance of the particles from the cluster center (in Mpc).

    - **mass:** *\[float ndarray\]*
        
        Mass of the particles (in solar Masses).

    - **Delta:** *\[float\]*
        
        Overdensity at which to compute the radius rDelta.

    - **z:** *\[float, optional\], default = 0.0*
        
        Redshift of the halo.

    - **H0:** *\[float, optional\], default = 67.5*
        
        Value of H0.

    - **Om:** *\[float, optional\], default = 0.32*
        
        Value of Omega matter.

    - **Ol:** *\[float, optional\], default = 0.68*
        
        Value of Omega Lambda.

    #### **Returns:**
    
    - **rDelta:** *\[float\]*

        The value of the radius enclosing Delta times the critical density in Mpc.
