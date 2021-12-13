import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table #, QTable, hstack, vstack
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.constants import G
#from astropy.coordinates import frame_transform_graph
#from astropy.coordinates.matrix_utilities import matrix_transpose

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic
from gala.dynamics.nbody import DirectNBody
#import galstreams

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy import interpolate

#from scipy.interpolate import InterpolatedUnivariateSpline

import pickle
import emcee
from multiprocessing import Pool
import corner


ham = gp.Hamiltonian(gp.MilkyWayPotential())
ham_bovy = gp.Hamiltonian(gp.BovyMWPotential2014())
ham_heavy = gp.Hamiltonian(gp.MilkyWayPotential(nucleus=dict(m=0), halo=dict(c=0.95, m=7e11), bulge=dict(m=4e9), disk=dict(m=5.5e10)))

tot_mass = 5.4e11
f_gse = 0.2
mw_mass = tot_mass * (1 - f_gse)
gse_mass = tot_mass * f_gse

mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.16, b=1.14, c=1, units=galactic, R=R.from_euler('xyz', (47, 10, 13), degrees=True))
ham_tilt = gp.Hamiltonian(mw_part + gse)

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()

#from superfreq import SuperFreq
import time

def subsample_sim(nskip=10):
    """"""
    
    t = Table(fits.getdata('../data/GSE_DM.fits'))
    t = t[::nskip]
    t.write('../data/GSE_DM_{:04d}.fits.gz'.format(nskip), overwrite=True)

def sim_potential(nskip=1000):
    """Fit NFW potential to the potential of dark matter particles from the GSE model"""
    
    t = Table(fits.getdata('../data/GSE_DM_{:04d}.fits.gz'.format(nskip)))
    c = coord.Galactocentric(x=t['X_gal']*u.kpc, y=t['Y_gal']*u.kpc, z=t['Z_gal']*u.kpc, v_x=t['Vx_gal']*u.km/u.s, v_y=t['Vy_gal']*u.km/u.s, v_z=t['Vz_gal']*u.km/u.s)
    
    N = 100
    cx = coord.Galactocentric(z=np.linspace(0,150,N)*u.kpc, y=np.zeros(N)*u.kpc, x=np.zeros(N)*u.kpc, v_x=np.zeros(N)*u.km/u.s, v_y=np.zeros(N)*u.km/u.s, v_z=np.zeros(N)*u.km/u.s)
    wx = gd.PhaseSpacePosition(cx.cartesian)
    
    c_arr = np.array([c.x.value, c.y.value, c.z.value]).T
    cx_arr = np.array([cx.x.value, cx.y.value, cx.z.value]).T
    
    dist = cdist(c_arr, cx_arr) * u.kpc
    dist_inv = dist**-1
    
    mdm = 1e4*u.Msun * nskip
    phi = (-G*mdm * np.sum(dist_inv, axis=0)).to(u.kpc**2 * u.Myr**-2)
    
    x = [np.log10(6e10), 16]
    
    res = minimize(lambda x: np.sum((gp.NFWPotential(m=10**x[0], r_s=x[1], units=galactic).energy(wx[1:]).value - phi[1:].value)**2), [np.log10(6e10), 16])
    print('m_s={:.3g} r_s={:.2f}'.format(10**res.x[0], x[1]))
    
    plt.close()
    plt.figure()
    
    plt.plot(cx.z, phi, 'ko', label='GSE DM')

    #nfw = gp.NFWPotential(m=5.8e10, r_s=9.8, units=galactic)
    nfw = gp.NFWPotential(m=10**res.x[0], r_s=res.x[1], units=galactic)
    plt.plot(cx.z, nfw.energy(wx), 'r-', label='NFW fit\n$m_s$={:.3g}M$_\odot$, $r_s$={:.2f}kpc'.format(10**res.x[0], res.x[1]))
    
    plt.xlabel('Z [kpc]')
    plt.ylabel('$\Phi(Z)$ [kpc$^2$ Myr$^{-2}$]')
    plt.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig('../plots/gse_potential_fit.png')

def dm_inertia(nskip=1000):
    """"""
    
    if nskip==1:
        t = Table(fits.getdata('../data/GSE_DM.fits'))
    else:
        t = Table(fits.getdata('../data/GSE_DM_{:04d}.fits.gz'.format(nskip)))
    c = coord.Galactocentric(x=t['X_gal']*u.kpc, y=t['Y_gal']*u.kpc, z=t['Z_gal']*u.kpc, v_x=t['Vx_gal']*u.km/u.s, v_y=t['Vy_gal']*u.km/u.s, v_z=t['Vz_gal']*u.km/u.s)
    ind = np.sqrt(c.x**2 + c.y**2 + c.z**2) < 50*u.kpc
    c = c[ind]
    
    m = np.array([c.x, c.y, c.z])
    print(np.median(m, axis=1))
    
    cv = np.cov(m)
    w, v = np.linalg.eig(cv)
    print(w/np.min(w))
    print(v)
    
    print(v[1])
    caxes = coord.Galactocentric(x=v[:,0]*u.kpc, y=v[:,1]*u.kpc, z=v[:,2]*u.kpc)
    caxes.representation_type = 'spherical'
    print(caxes)
    
    #print(np.arcsin(v[:,2]))
    #print(np.arccos(v[:,2]))
    
    #r, lat, lon = coord.cartesian_to_spherical(v[:,0], v[:,1], v[:,2])
    #print(lon, lat)

def gse_fit(nskip=1000):
    """Fit NFW halo to GSE DM particles"""
    
    t = Table(fits.getdata('../data/GSE_DM_{:04d}.fits.gz'.format(nskip)))
    c = coord.Galactocentric(x=t['X_gal']*u.kpc, y=t['Y_gal']*u.kpc, z=t['Z_gal']*u.kpc, v_x=t['Vx_gal']*u.km/u.s, v_y=t['Vy_gal']*u.km/u.s, v_z=t['Vz_gal']*u.km/u.s)
    
    np.random.seed(138)
    
    N = 500
    rmax = 300
    pos = (np.random.rand(3,N)*2*rmax - rmax) * u.kpc
    vel = np.zeros((3,N)) * u.km/u.s
    cx = coord.Galactocentric(x=pos[0], y=pos[1], z=pos[2], v_x=vel[0], v_y=vel[1], v_z=vel[2])
    wx = gd.PhaseSpacePosition(cx.cartesian)
    
    c_arr = np.array([c.x.value, c.y.value, c.z.value]).T
    cx_arr = np.array([cx.x.value, cx.y.value, cx.z.value]).T
    
    dist = cdist(c_arr, cx_arr) * u.kpc
    dist_inv = dist**-1
    
    mdm = 1e4*u.Msun * nskip
    phi = (-G*mdm * np.sum(dist_inv, axis=0)).to(u.kpc**2 * u.Myr**-2)
    
    #p0 = [np.log10(9e10), 16]
    #res = minimize(lambda x: np.sum((gp.NFWPotential(m=10**x[0], r_s=x[1], units=galactic).energy(wx).value - phi.value)**2/(0.15*phi.value)**2), p0)
    
    p0 = [np.log10(9e10), 16, 1.1, 1.05]
    res = minimize(lambda x: np.sum((gp.NFWPotential(m=10**x[0], r_s=x[1], a=x[2], b=x[3], c=1., units=galactic).energy(wx).value - phi.value)**2/(0.15*phi.value)**2), p0)
    
    #p0 = [np.log10(9e10), 16, 1., 1., 0., 0., 0.]
    #res = minimize(lambda x: np.sum((gp.NFWPotential(m=10**x[0], r_s=x[1], a=x[2], b=x[3], c=1., units=galactic, R=R.from_euler('xyz', (x[4], x[5], x[6]), degrees=True)).energy(wx).value - phi.value)**2/(0.15*phi.value)**2), p0)
    
    print('m_s={:.3g} r_s={:.2f}'.format(10**res.x[0], res.x[1]))
    print(res)
    
    phi_fit = gp.NFWPotential(m=10**res.x[0], r_s=res.x[1], units=galactic).energy(wx)
    residuals = np.array(1 - (phi_fit/phi).decompose())
    
    #print(residuals)
    plt.close()
    plt.figure()
    
    plt.hist(residuals, bins=30)
    
    plt.tight_layout()


def initial_mass(M, t, rapo, vcirc, ecc):
    """Estimates initial mass of a globular cluster,
    accounting for mass loss due to stellar evolution and tidal dissolution
    following Lamers et al. (2005)
    
    Parameters
    ----------
    M : :class:`~astropy.units.Quantity`
    Current mass of the cluster
    t : :class:`~astropy.units.Quantity`
    Age of the cluster (or rather, how long it has been dissolving)
    rapo : :class:`~astropy.units.Quantity`
    Orbital apocenter
    vcirc : :class:`~astropy.units.Quantity`
    Circular velocity at the apocenter
    ecc : float
    Orbital eccentricity
    """
    
    # appropriate for Z = 0.02 Zsun, Kruijssen & Lamers (2008)
    a_ev = 6.93
    b_ev = 0.255
    c_ev = -1.682
    
    log_q_ev = (np.log10(t.to(u.yr).value) - a_ev)**b_ev + c_ev
    mu_ev = 1 - 10**log_q_ev

    
    # Kruijssen & Mieske (2009), Lamers et al. (2010)
    gamma = 0.7
    t0_sun = 10.7*u.Myr
    
    t0 = t0_sun * (rapo/(8.5*u.kpc)) * (vcirc/(220*u.km/u.s))**-1 * (1-ecc)

    
    # Lamers et al. (2005) Eq 7
    Mi = ((M.to(u.Msun).value**gamma + (gamma*t/t0).decompose())**(1/gamma) * mu_ev**-1) * u.Msun
    
    
    return Mi

def mass():
    """"""
    t = Table.read('../data/gc_vasiliev.fits')

    ind = [True if 'NGC 5824' in name else False for name in t['Name']]
    tc = t[ind]
    tc.pprint()
    
    c = coord.SkyCoord(ra=tc['RAdeg'], dec=tc['DEdeg'], distance=tc['Dist'], pm_ra_cosdec=tc['pmRA'], pm_dec=tc['pmDE'], radial_velocity=tc['HRV'], frame='icrs')
    cg = c.transform_to(coord.Galactocentric)
    
    w0 = gd.PhaseSpacePosition(cg[0].cartesian)
    orbit = ham.integrate_orbit(w0, dt=-1*u.Myr, n_steps=10000)

    rapo = orbit.apocenter()
    ecc = orbit.eccentricity()
    vcirc = ham.potential.circular_velocity(np.array([rapo.to(u.kpc).value, 0, 0]))[0]
    
    # Present-day mass []
    M = 7.6e5*u.Msun

    # Assume stream is 5 Gyr old
    T = 5*u.Gyr

    Mi = initial_mass(M, T, rapo, vcirc, ecc)
    
    return Mi

def coords_triangulum():
    """"""
    
    pkl = pickle.load(open('../data/data_triangulum.pkl', 'rb'))
    c = coord.SkyCoord(ra=pkl['dec'][0], dec=pkl['dec'][1], distance=pkl['dist'][1], pm_ra_cosdec=pkl['pmra'][1], pm_dec=pkl['pmdec'][1], radial_velocity=pkl['vr'][1], frame='icrs')
    print(c)
    
    pickle.dump(c, open('../data/coords_triangulum.pkl', 'wb'))
    
    cg = c.transform_to(coord.Galactic)
    c_xy = [cg.l, cg.b, c.distance, c.pm_ra_cosdec, c.pm_dec, c.radial_velocity]
    pickle.dump(c_xy, open('../data/figcoords_triangulum.pkl', 'wb'))

    
def coords_4d(name):
    """"""
    
    pkl = pickle.load(open('../data/data_{:s}.pkl'.format(name), 'rb'))
    
    for c in ['dec', 'dist', 'pmra', 'pmdec']:
        print(len(pkl[c][0]))
    
    c = coord.SkyCoord(ra=pkl['dec'][0], dec=pkl['dec'][1], pm_ra_cosdec=pkl['pmra'][1], pm_dec=pkl['pmdec'][1], frame='icrs')
    pickle.dump(c, open('../data/coords_{:s}.pkl'.format(name), 'wb'))
    
    cg = c.transform_to(coord.Galactic)
    c_xy = [cg.l, cg.b, c.ra*np.nan, c.pm_ra_cosdec, c.pm_dec, c.ra*np.nan]
    pickle.dump(c_xy, open('../data/figcoords_{:s}.pkl'.format(name), 'wb'))

def coords_4d_b(name):
    """"""
    
    pkl = pickle.load(open('../data/data_{:s}.pkl'.format(name), 'rb'))
    
    for c in ['dec', 'dist', 'pmra', 'pmdec']:
        print(len(pkl[c][0]))
    
    c = coord.SkyCoord(ra=pkl['dec'][0], dec=pkl['dec'][1], pm_ra_cosdec=pkl['pmra'][1], pm_dec=pkl['pmdec'][1], frame='icrs')
    pickle.dump(c, open('../data/coords_{:s}.pkl'.format(name), 'wb'))
    
    cg = c.transform_to(coord.Galactic)
    c_xy = [cg.b, cg.l, c.ra*np.nan, c.pm_ra_cosdec, c.pm_dec, c.ra*np.nan]
    pickle.dump(c_xy, open('../data/figcoords_{:s}.pkl'.format(name), 'wb'))

    
def stream(name='ngc5824', f=0.16, nskip=20):
    """"""
    
    clusters = {}
    clusters['ngc5824'] = dict(name='NGC 5824', mass=7.6e5*u.Msun, r=6.5*u.pc, age=5*u.Gyr, wangle=75*u.deg, fl=True, streams=['triangulum', 'turbio'])
    clusters['ngc5272'] = dict(name='NGC 5272', mass=4.1e5*u.Msun, r=6.34*u.pc, age=0.5*u.Gyr, wangle=180*u.deg, fl=False, streams=['svol',])
    clusters['ngc5024'] = dict(name='NGC 5024', mass=4.6e5*u.Msun, r=10.18*u.pc, age=5*u.Gyr, wangle=180*u.deg, fl=True, streams=['ravi',])
    clusters['ngc3201'] = dict(name='NGC 3201', mass=1.6e5*u.Msun, r=6.78*u.pc, age=2*u.Gyr, wangle=20*u.deg, fl=True, streams=[])
    clusters['ngc4590'] = dict(name='NGC 4590', mass=1.2e5*u.Msun, r=7.58*u.pc, age=2*u.Gyr, wangle=20*u.deg, fl=False, streams=[])
    clusters['ngc5139'] = dict(name='NGC 5139', mass=3.64e6*u.Msun, r=10.36*u.pc, age=0.5*u.Gyr, wangle=180*u.deg, fl=True, streams=[])
    
    cluster = clusters[name]
    
    t = Table.read('../data/gc_vasiliev.fits')
    ind = [True if cluster['name'] in name_ else False for name_ in t['Name']]
    tc = t[ind]
    
    c = coord.SkyCoord(ra=tc['RAdeg'], dec=tc['DEdeg'], distance=tc['Dist'], pm_ra_cosdec=tc['pmRA'], pm_dec=tc['pmDE'], radial_velocity=tc['HRV'], frame='icrs')
    g = c.transform_to(coord.Galactic)
    rep = c.transform_to(coord.Galactocentric).data
    w0 = gd.PhaseSpacePosition(rep)
    
    gc_mass = cluster['mass']
    rs = np.random.RandomState(5134)
    df = ms.FardalStreamDF(random_state=rs)
    
    gc_pot = gp.PlummerPotential(m=gc_mass, b=cluster['r'], units=galactic)
    mw = gp.MilkyWayPotential()
    #print(mw.values())
    
    tot_mass = 5.4e11
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.28, b=1.3/1.28, c=1, units=galactic, R=R.from_euler('yz',(-35,-35), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.4, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, -120), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=14.5, a=1.07, b=1.05, c=1, units=galactic, R=R.from_euler('xyz', (42, 7, 63), degrees=True))
    mw_tilt = mw_part + gse
    
    #mw = gp.MilkyWayPotential(halo={'m': tot_mass})

    ## From Rohan
    #mw = gp.MilkyWayPotential(units=galactic)
    #gse = gp.NFWPotential(m=1e11, r_s=10, a=1.5, b=1, c=1, units=galactic, R=R.from_euler(‘yz’,(-35,-30),degrees=True))
    #pot = mw + gse
    
    dt = -1*u.Myr
    n_steps = np.int(np.abs((cluster['age']/dt).decompose()))
    
    gen_stream = ms.MockStreamGenerator(df, mw, progenitor_potential=gc_pot)
    stream, _ = gen_stream.run(w0, gc_mass, dt=dt, n_steps=n_steps, release_every=nskip)
    stream_c = stream.to_coord_frame(coord.ICRS)
    stream_g = stream.to_coord_frame(coord.Galactic)
    
    gen_stream_tilt = ms.MockStreamGenerator(df, mw_tilt, progenitor_potential=gc_pot)
    stream_tilt, _ = gen_stream_tilt.run(w0, gc_mass, dt=dt, n_steps=n_steps, release_every=nskip)
    stream_tilt_c = stream_tilt.to_coord_frame(coord.ICRS)
    stream_tilt_g = stream_tilt.to_coord_frame(coord.Galactic)

    wangle = cluster['wangle']
    
    plt.close()
    fig, ax = plt.subplots(5,1,figsize=(16,16), sharex=True)
    
    obs = []
    for s in cluster['streams']:
        pkl = pickle.load(open('../data/figcoords_{:s}.pkl'.format(s), 'rb'))
        obs += [pkl]
    
    if cluster['fl']:
        stream_xy = [stream_g.l.wrap_at(wangle), stream_g.b, stream_g.distance, stream_c.pm_ra_cosdec, stream_c.pm_dec, stream_g.radial_velocity]
        stream_tilt_xy = [stream_tilt_g.l.wrap_at(wangle), stream_tilt_g.b, stream_tilt_g.distance, stream_tilt_c.pm_ra_cosdec, stream_tilt_c.pm_dec, stream_tilt_g.radial_velocity]
        cluster_xy = [g.l.wrap_at(wangle), g.b, c.distance, c.pm_ra_cosdec, c.pm_dec, c.radial_velocity]
        for j in range(len(obs)):
            obs[j][0] = obs[j][0].wrap_at(wangle)
    
        labels = ['l [deg]', 'b [deg]', 'Distance [kpc]', '$\mu_\\alpha$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    
    else:
        stream_xy = [stream_g.b, stream_g.l.wrap_at(wangle), stream_g.distance, stream_c.pm_ra_cosdec, stream_c.pm_dec, stream_g.radial_velocity]
        stream_tilt_xy = [stream_tilt_g.b, stream_tilt_g.l.wrap_at(wangle), stream_tilt_g.distance, stream_tilt_c.pm_ra_cosdec, stream_tilt_c.pm_dec, stream_tilt_g.radial_velocity]
        cluster_xy = [g.b, g.l.wrap_at(wangle), c.distance, c.pm_ra_cosdec, c.pm_dec, c.radial_velocity]
        for j in range(len(obs)):
            obs[j][1] = obs[j][1].wrap_at(wangle)
    
        labels = ['b [deg]', 'l [deg]', 'Distance [kpc]', '$\mu_\\alpha$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    
    Nstream = np.size(stream_xy[0])
    colors = ['tab:blue', 'tab:green']
    
    for i in range(5):
        plt.sca(ax[i])
        plt.plot(stream_xy[0], stream_xy[i+1], 'k.', ms=1, label='Model [fiducial]')
        plt.plot(stream_tilt_xy[0], stream_tilt_xy[i+1], 'r.', ms=1, label='Model [GSE f={:g}]'.format(f))
        
        #plt.scatter(stream_xy[0], stream_xy[i+1], c=np.linspace(0,Nstream-1,Nstream), s=5, label='Model [fiducial]')

        plt.plot(cluster_xy[0], cluster_xy[i+1], 'H', mew=2, mec='cyan', mfc='none', ms=15, label='NGC 5824')
        
        for j in range(len(obs)):
            plt.plot(obs[j][0].wrap_at(wangle), obs[j][i+1], '^', color=colors[j], ms=5, label='{:s}'.format(cluster['streams'][j]))
        
        plt.ylabel(labels[i+1])
    
    plt.xlabel(labels[0])
    
    plt.sca(ax[0])
    plt.legend(fontsize='small', loc=0)
    
    plt.tight_layout()
    plt.savefig('../plots/{:s}_gse_f{:.2f}.png'.format(name, f))
    

def save_models_nskip():
    """Test how many stars are needed for a reliable stream track"""
    
    name = 'ngc5824'
    cluster = dict(name='NGC 5824', mass=7.6e5*u.Msun, r=6.5*u.pc, age=5*u.Gyr, wangle=75*u.deg, fl=True, streams=['triangulum', 'turbio'])

    t = Table.read('../data/gc_vasiliev.fits')
    ind = [True if cluster['name'] in name_ else False for name_ in t['Name']]
    tc = t[ind]
    
    c = coord.SkyCoord(ra=tc['RAdeg'], dec=tc['DEdeg'], distance=tc['Dist'], pm_ra_cosdec=tc['pmRA'], pm_dec=tc['pmDE'], radial_velocity=tc['HRV'], frame='icrs')
    g = c.transform_to(coord.Galactic)
    rep = c.transform_to(coord.Galactocentric).data
    w0 = gd.PhaseSpacePosition(rep)
    
    gc_mass = cluster['mass']
    rs = np.random.RandomState(5134)
    df = ms.FardalStreamDF(random_state=rs)
    
    gc_pot = gp.PlummerPotential(m=gc_mass, b=cluster['r'], units=galactic)
    mw = gp.MilkyWayPotential()
    
    dt = -1*u.Myr
    n_steps = np.int(np.abs((cluster['age']/dt).decompose()))
    
    gen_stream = ms.MockStreamGenerator(df, mw, progenitor_potential=gc_pot)
    
    for nskip in [50,20,10,5,1]:
        t1 = time.time()
        stream, _ = gen_stream.run(w0, gc_mass, dt=dt, n_steps=n_steps, release_every=nskip)
        stream_c = stream.to_coord_frame(coord.ICRS)
        t2 = time.time()
        
        pickle.dump(stream_c, open('../data/mockstream_{:s}_{:03d}.pkl'.format(name, nskip), 'wb'))
        print(nskip, n_steps/nskip, t2-t1)

def plot_models_nskip(name='ngc5824'):
    """"""
    
    stream = []
    wangle = 90*u.deg
    
    plt.close()
    plt.figure(figsize=(15,15))
    
    for e, nskip in enumerate([1, 5, 10, 20, 50]):
        stream += [pickle.load(open('../data/mockstream_{:s}_{:03d}.pkl'.format(name, nskip), 'rb'))]
        
        stream_g = stream[e].transform_to(coord.Galactic)
        
        isort = np.argsort(stream_g.l.wrap_at(wangle))
        
        #plt.plot(stream_g.l.wrap_at(wangle)[isort], stream_g.b[isort], '-')
        plt.plot(stream_g.l.wrap_at(wangle), stream_g.b, 'o')
        
    plt.tight_layout()

def interpolate_track(name='ngc5824', nskip=1):
    """"""
    
    stream = pickle.load(open('../data/mockstream_{:s}_{:03d}.pkl'.format(name, nskip), 'rb'))
    stream_g = stream.transform_to(coord.Galactic)
    wangle = 90*u.deg
    
    isort = np.argsort(stream_g.l.wrap_at(wangle))
    x = stream_g.l.wrap_at(wangle).value[isort]
    y = stream_g.b.value[isort]
    #s = interpolate.InterpolatedUnivariateSpline(x[isort], y[isort])
    t = np.linspace(-220, 50, 20)
    s = interpolate.LSQUnivariateSpline(x, y, t, k=3)
    
    xnew = np.linspace(-250, 85, 1000)
    ynew = s(xnew)
    
    plt.close()
    plt.figure(figsize=(15,15))
    
    plt.plot(x, y - s(x), 'o')
    #plt.plot(xnew, ynew, '-')
    
    plt.tight_layout()

def nskip_interpolation(name='ngc5824'):
    """"""
    
    wangle = 90*u.deg
    t = np.linspace(-220, 50, 20)
    k = 3
    
    streams = []
    nskips = [1, 5, 10, 20]
    
    for e, nskip in enumerate(nskips):
        stream = pickle.load(open('../data/mockstream_{:s}_{:03d}.pkl'.format(name, nskip), 'rb'))
        stream_g = stream.transform_to(coord.Galactic)
        isort = np.argsort(stream_g.l.wrap_at(wangle))
        x = stream_g.l.wrap_at(wangle).value[isort]
        y = stream_g.b.value[isort]
        s = interpolate.LSQUnivariateSpline(x, y, t, k=k)
        
        dict_ = dict(stream=stream, stream_g=stream_g, x=x, y=y, s=s)
        streams += [dict_]
    
    xnew = np.linspace(-250, 85, 1000)
    ynew = streams[0]['s'](xnew)
    
    plt.close()
    plt.figure(figsize=(16,8))
    
    plt.plot(streams[0]['x'], streams[0]['y'] - streams[0]['s'](streams[0]['x']), 'k.', alpha=0.3)
    
    for i in range(len(streams)):
        plt.plot(xnew, streams[i]['s'](xnew) - ynew, '-', label='Nskip = {:d}'.format(nskips[i]))
    
    plt.legend()
    plt.ylim(-2,2)
    #plt.gca().set_aspect('equal')
    
    plt.tight_layout()


def gap_dist(f=0.167, rs=16, q_a=1.3, q_b=1, q_c=1, theta_y=-35, theta_z=-35, verbose=False):
    """Calculate distance between the GD-1 gap and globular clusters in a tilted halo potential"""
    
    # Read in gaps' 6D position
    cgap = pickle.load(open('../data/gap40_location.pkl', 'rb'))
    g_gap = cgap.transform_to(coord.Galactic)
    rep_gap = cgap.transform_to(coord.Galactocentric).data
    w0_gap = gd.PhaseSpacePosition(rep_gap)
    
    # Read in globular cluster 6D positions
    t = Table.read('../data/gc_vasiliev.fits')
    c = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    g = c.transform_to(coord.Galactic)
    rep = c.transform_to(coord.Galactocentric).data
    w0 = gd.PhaseSpacePosition(rep)
    
    # set up the tilted halo potential
    tot_mass = 5.4e11
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    #gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=q_a, b=q_b, c=q_c, units=galactic, R=R.from_euler('yz', (theta_y, theta_z), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1, c=1, units=galactic, R=R.from_euler('xyz', (120, 60, -30), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, 30), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.4, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, -120), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=14.5, a=1.07, b=1.05, c=1, units=galactic, R=R.from_euler('xyz', (42, 7, 63), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.16, b=1.14, c=1, units=galactic, R=R.from_euler('xyz', (47, 10, 13), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    mw_tilt = mw_part + gse
    
    # set up time integration
    dt = -0.5*u.Myr
    T = 5*u.Gyr
    nstep = np.int(np.abs(T/dt).decompose())
    
    # gap orbit
    orbit_gap = mw_tilt.integrate_orbit(w0_gap, dt=dt, n_steps=nstep)
    pos_gap = np.array([orbit_gap.pos.x, orbit_gap.pos.y, orbit_gap.pos.z])
    
    # globular cluster orbits
    orbits = []
    Ngc = len(t)
    
    for i in range(Ngc):
        orbit_gc = mw_tilt.integrate_orbit(w0[i], dt=dt, n_steps=nstep)
        orbits += [orbit_gc]
        
    plt.close()
    plt.figure(figsize=(12,7))
    
    for i in range(Ngc):
        pos = np.array([orbits[i].pos.x, orbits[i].pos.y, orbits[i].pos.z])
        dist = np.linalg.norm(pos_gap - pos, axis=0)
        
        plt.plot(orbit_gap.t, dist, 'k-', alpha=0.3, label='')
        
        if np.any(dist<1.6):
            if verbose:
                print('{:s} dmin={:.2f} Etot={:.3f} Lz={:.2f} Lperp={:.2f} ecc={:.2f} rperi={:.2f} rapo={:.2f} zmax={:.2f}'.format(t['Name'][i], np.min(dist), orbits[i].energy()[0], orbits[i].angular_momentum()[2,0], np.sqrt(orbits[i].angular_momentum()[0,0]**2 + orbits[i].angular_momentum()[1,0]**2), orbits[i].eccentricity(), orbits[i].pericenter(), orbits[i].apocenter(), orbits[i].zmax()))
            plt.plot(orbit_gap.t, dist, '-', zorder=200, label=t['Name'][i])
    
    plt.gca().set_yscale('log')
    plt.minorticks_on()
    plt.xlim(-5000,0)
    plt.ylim(5e-2,3e2)
    
    plt.xlabel('Time [Myr]')
    plt.ylabel('Distance [kpc]')
    plt.legend(ncol=3, loc=2, fontsize='small')
    #plt.title('GSE f = {:.3f} $r_s$ = {:.1f} $\\theta_y$ = {:.0f} $\\theta_z$ = {:.0f}'.format(f, rs, theta_y, theta_z), fontsize='medium')
    plt.title('GSE f = {:.3f} $r_s$ = {:.1f}'.format(f, rs), fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/gap40_distances_f.{:.3f}_rs.{:.1f}.png'.format(f, rs))

def rotmat():
    """"""
    theta_y = -35
    theta_z = -35
    r = R.from_euler('yz', (theta_y, theta_z), degrees=True)
    
    # Tait-Bryan angles, extrensic rotation 'xyz', with x=0
    
    print(r.as_matrix())
    #print(r.as_euler('xyz', degrees=True))
    
def ln_likelihood(x, w0, w0_gap, f=0.167, rs=16, q_a=1.3, q_b=1, q_c=1, theta_y=-35, theta_z=-35, save_blob=True):
    """"""
    
    # set up time integration
    dt = -0.5*u.Myr
    T = 2.5*u.Gyr
    T = 1.*u.Gyr
    nstep = np.int(np.abs(T/dt).decompose())
    #half_nstep = np.int(nstep*0.5)
    
    # potential parameters
    f, rs, q_a, q_b = x
    
    ## normalize shape to conserve volume?
    #qtot = 1.3
    #q_b = qtot/q_a
    
    if (f<0) | (f>1) | (rs<0) | (rs>100) | (q_a<0.5) | (q_a>2) | (q_b<0.5) | (q_b>2) | (q_c<0.5) | (q_c>2):
        if save_blob:
            return -np.inf, np.empty(nstep+1)*np.nan
        else:
            return -np.inf
    
    # set up the tilted halo potential
    tot_mass = 5.4e11
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=q_a, b=q_b, c=q_c, units=galactic, R=R.from_euler('yz', (theta_y, theta_z), degrees=True))
    mw_tilt = mw_part + gse
    
    # gap orbit
    orbit_gap = mw_tilt.integrate_orbit(w0_gap, dt=dt, n_steps=nstep)
    pos_gap = np.array([orbit_gap.pos.x, orbit_gap.pos.y, orbit_gap.pos.z])
    
    # cluster orbit
    orbit = mw_tilt.integrate_orbit(w0, dt=dt, n_steps=nstep)
    pos = np.array([orbit.pos.x, orbit.pos.y, orbit.pos.z])
    
    dist = np.linalg.norm(pos_gap - pos, axis=0)
    #dist = np.linalg.norm(pos_gap[:,3000:4000] - pos[:,3000:4000], axis=0)
    
    # 1851
    i0 = 3150
    i1 = 4550
    
    # 4590
    i0 = 0
    i1 = -1
    
    dmin = np.min(dist[i0:i1]) + 1e-10
    sigma = 0.02
    chi2 = - (dmin/sigma)**2
    
    #isort = np.argsort(dist[i0:i1])
    #dmin = np.sum(dist[i0:i1][isort][:100])

    blob = dist
    
    #if dmin>5:
        #return -np.inf, blob
    #else:
        #return -3*np.log(dmin + 1e-10), blob
    
    if save_blob:
        return chi2, blob
    else:
        return chi2
    
def fit_gse(name='NGC 1851', full=False, save_blob=False, theta_z=-35):
    """"""
    savename = name.lower().replace(' ', '')

    # Read in gap 6D position
    cgap = pickle.load(open('../data/gap_location.pkl', 'rb'))
    g_gap = cgap.transform_to(coord.Galactic)
    rep_gap = cgap.transform_to(coord.Galactocentric).data
    w0_gap = gd.PhaseSpacePosition(rep_gap)
    
    # Read in globular cluster 6D positions
    t = Table.read('../data/gc_vasiliev.fits')
    ind = [True if name in name_ else False for name_ in t['Name']]
    t = t[ind]
    c = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    g = c.transform_to(coord.Galactic)
    rep = c.transform_to(coord.Galactocentric).data
    w0 = gd.PhaseSpacePosition(rep)
    
    #p0s = np.array([0.16, 15.5, 1.2, 0.9])
    #p0s = np.array([0.17, 16, 1.3]) # 1851
    p0s = np.array([0.16, 20.5, 0.9, 0.9]) # 4590
    res = minimize(lambda *x: -ln_likelihood(*x, save_blob=False), x0=p0s, args=(w0, w0_gap))
    print(res.x)
    
    print(ln_likelihood(p0s, w0, w0_gap, save_blob=False))
    print(ln_likelihood(res.x, w0, w0_gap, save_blob=False))
    
    if full:
        nth = 3
        seed = 5736
        nwalkers = 64
        nsteps = 512
        
        pool = Pool(nth)
        np.random.seed(seed)
        p0 = emcee.utils.sample_ball(p0s, [1e-3, 1e-3, 1e-3, 1e-3], nwalkers)
        p0 = np.abs(p0)
        
        sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1], log_prob_fn=ln_likelihood, pool=pool, args=(w0, w0_gap), kwargs=dict(save_blob=save_blob, theta_z=theta_z))
        _ = sampler.run_mcmc(p0, nsteps)
        
        pickle.dump(sampler, open('../data/fits/mcmc_{:s}_{:d}_b{:d}.pkl'.format(savename, sampler.ndim, save_blob), 'wb'))
        print(np.median(sampler.flatchain, axis=0))
        
        pool.close()


def plot_chains(name='ngc1851', npar=1, save_blob=False):
    """Plot chain"""
    
    sampler = pickle.load(open('../data/fits/mcmc_{:s}_{:d}_b{:d}.pkl'.format(name, npar, save_blob), 'rb'))
    
    names = [r'f', r'$r_s$', r'$q_a$', r'$q_b$']
    
    plt.close()
    fig, ax = plt.subplots(sampler.ndim, 1, figsize=(10,10), sharex=True, squeeze=False)

    for k in range(sampler.ndim):
        for walker in sampler.chain[..., k]:
            ax[k][0].plot(walker, marker='', drawstyle='steps-mid', alpha=0.2)
        ax[k][0].set_ylabel(names[k])
    
    plt.sca(ax[sampler.ndim-1][0])
    plt.xlabel('Step')

    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/mcmc_chains_{:s}_{:d}.png'.format(name, npar))

def plot_blobs(name='ngc1851', npar=1, save_blob=False):
    """Plot blobs"""
    
    sampler = pickle.load(open('../data/fits/mcmc_{:s}_{:d}_b{:d}.pkl'.format(name, npar, save_blob), 'rb'))
    blobs = sampler.get_blobs()
    lnprob = sampler.get_log_prob()
    #print(np.shape(lnprob))
    #print(np.percentile(lnprob, [5,50,95]))
    
    chain = sampler.get_chain()
    
    imin = np.argmin(blobs[:,:,3150:4550], axis=2)
    dmin = np.min(blobs[:,:,3150:4550], axis=2)
    
    plt.close()
    plt.figure()
    
    #plt.scatter(blobs[:,:,0], blobs[:,:,1], c=lnprob, s=3)
    #plt.scatter(blobs[:,:,0], blobs[:,:,1], c=chain, s=3)
    plt.scatter(imin, dmin, c=chain, s=3)
    
    plt.xlabel('Time index')
    plt.ylabel('Minimum distance [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/mcmc_blobs_{:s}_{:d}.png'.format(name, npar))
    
def plot_distblob(name='ngc1851', npar=1, save_blob=False):
    """"""
    sampler = pickle.load(open('../data/fits/mcmc_{:s}_{:d}_b{:d}.pkl'.format(name, npar, save_blob), 'rb'))
    blobs = sampler.get_blobs(flat=True, discard=256, thin=100)
    lnprob = sampler.get_log_prob(flat=True, discard=256, thin=100)
    
    print(np.shape(blobs), np.shape(lnprob), np.min(lnprob), np.max(lnprob))
    N = np.size(lnprob)
    
    plt.close()
    plt.figure(figsize=(15,7))
    
    for i in range(N):
        plt.plot(blobs[i], '-', alpha=0.1)
    
    plt.gca().set_yscale('log')
    plt.xlabel('Time step')
    plt.ylabel('Distance [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/mcmc_distance_{:s}_{:d}.png'.format(name, npar))
    
def chain_median(name='ngc1851', npar=1, save_blob=False):
    """Output best estimate"""
    
    sampler = pickle.load(open('../data/fits/mcmc_{:s}_{:d}_b{:d}.pkl'.format(name, npar, save_blob), 'rb'))
    chain = sampler.get_chain(flat=True, discard=256)
    
    print(np.shape(chain))
    print(np.percentile(chain, [16,50,84], axis=0))

def plot_corner(name='ngc1851', npar=1, save_blob=False, bins=30):
    """"""
    
    sampler = pickle.load(open('../data/fits/mcmc_{:s}_{:d}_b{:d}.pkl'.format(name, npar, save_blob), 'rb'))
    chain = sampler.get_chain(flat=True, discard=256)
    
    names = [r'f', r'$r_s$ [kpc]', r'$q_a$', r'$q_b$']
    names = names[:npar]
    
    plt.close()
    fig, ax = plt.subplots(npar, npar, figsize=(10,10))
    corner.corner(chain, bins=bins, labels=names, show_titles=True, title_fmt='.3f', title_kwargs=dict(fontsize='small'), fig=fig)

    plt.tight_layout(h_pad=0.1,w_pad=0.1)
    plt.savefig('../plots/mcmc_corner_{:s}_{:d}.png'.format(name, npar))



def test_halo_mass():
    """"""
    
    f = 0.2
    tot_mass = 5.4e11
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse1 = gp.NFWPotential(m=gse_mass, r_s=16, a=0.8, b=1, c=1, units=galactic, R=R.from_euler('yz',(-35,-35), degrees=True))
    gse2 = gp.NFWPotential(m=gse_mass, r_s=16, a=1, b=1, c=1, units=galactic, R=R.from_euler('yz',(-35,-35), degrees=True))
    gse3 = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1, c=1, units=galactic, R=R.from_euler('yz',(-35,-35), degrees=True))
    gse4 = gp.NFWPotential(m=gse_mass, r_s=16, a=2, b=1, c=0.629, units=galactic, R=R.from_euler('yz',(-35,-35), degrees=True))
    #mw_tilt = mw_part + gse
    
    gse = [gse2, gse1, gse3, gse4]
    
    pos = np.array([0,0,400]) * u.kpc
    
    print(np.sqrt(2))
    
    for i in range(3):
        print(gse[i+1].mass_enclosed(pos)/gse[0].mass_enclosed(pos))



################
# GD-1 encounter

def orbit_section():
    """Store orbit section around the gap today (unperturbed stream particles)"""
    
    # stream particles unperturbed
    res = pickle.load(open('/home/ana/projects/disrupted_gc/data/fits/minimization_gd1.pkl', 'rb'))
    dec, dist, pmra, pmdec, vr = res.x
    ra = 123
    
    c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, radial_velocity=vr*u.km/u.s, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    T = 60*u.Myr
    nstep = 3000
    dt = -T/nstep
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=nstep)
    model_eq = orbit.to_coord_frame(coord.ICRS, galactocentric_frame=gc_frame)
    model = orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    ind = (model.phi1>-50*u.deg) & (model.phi1<-30*u.deg)
    print(np.sum(ind))
    
    pickle.dump(model[ind], open('../data/gd1_section_present.pkl','wb'))
    
    plt.close()
    plt.figure(figsize=(15,6))
    
    plt.plot(model.phi1, model.phi2, 'k-')
    plt.plot(model.phi1[ind], model.phi2[ind], 'ro')
    
    plt.tight_layout()

def encounter():
    """"""
    
    # Global gravitational potential
    f = 0.17
    f = 0.1
    rs = 15.769
    q_a = 1.3
    q_b = 1.
    q_c = 1.
    theta_y = -35
    theta_z = -35
    
    tot_mass = 5.4e11
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=q_a, b=q_b, c=q_c, units=galactic, R=R.from_euler('yz', (theta_y, theta_z), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, 30), degrees=True))
    gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.4, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, -120), degrees=True))
    mw_tilt = mw_part + gse
    
    
    # Cluster potential
    gc_mass = 3.18e5*u.Msun
    gc_rh = 2.9*u.pc
    gc_pot = gp.PlummerPotential(m=gc_mass, b=gc_rh, units=galactic)
    
    
    # NGC 1851 today
    t = Table.read('../data/gc_vasiliev.fits')
    name = 'NGC 1851'
    name = 'NGC 5904'
    ind = [True if name in name_ else False for name_ in t['Name']]
    t = t[ind]
    
    c = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_cluster = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    
    # read in GD-1 today
    stream_init = pickle.load(open('../data/gd1_section_present.pkl', 'rb'))
    w0_stream = gd.PhaseSpacePosition(stream_init.transform_to(gc_frame).cartesian)
    
    
    # integrate back in time
    Tback = 2*u.Gyr
    dt = -0.5*u.Myr
    Nback = np.abs(np.int((Tback/dt).decompose()))
    
    #orbit_cluster = mw_tilt.integrate_orbit(w0_cluster, dt=dt, n_steps=Nback)
    #w1_cluster = orbit_cluster[-1]
    
    #orbit_stream = mw_tilt.integrate_orbit(w0_stream, dt=dt, n_steps=Nback)
    #w1_stream = orbit_stream[-1,:]
    
    
    # run Nbody back to the present
    w0 = gd.combine((w0_cluster, w0_stream))
    particle_pot = [None] * (np.size(stream_init) + 1)
    nbody = DirectNBody(w0, particle_pot, external_potential=mw_tilt)
    orbits_back = nbody.integrate_orbit(dt=dt, n_steps=Nback)

    w1 = orbits_back[-1,:]
    particle_pot = [None] * (np.size(stream_init) + 1)
    particle_pot[0] = gc_pot
    nbody = DirectNBody(w1, particle_pot, external_potential=mw_tilt)
    orbits_fwd = nbody.integrate_orbit(dt=-dt, n_steps=Nback)
    
    print(w0_cluster)
    print(orbits_fwd[-1,0])
    
    
    
    #nbody = DirectNBody(w1, particle_pot, external_potential=mw_tilt)
    #orbits = nbody.integrate_orbit(dt=-dt, n_steps=Nback)
    ##orbits = mw_tilt.integrate_orbit(w1, dt=-dt, n_steps=Nback)
    
    ##pickle.dump(orbits.data, open('../data/encounter.pkl', 'wb'))
    #print(w0_cluster)
    #print(orbits[-1,0])
    
    out_dict = dict(pos=orbits_fwd.pos, vel=orbits_fwd.vel, t=orbits_fwd.t)
    pickle.dump(out_dict, open('../data/encounter.pkl', 'wb'))
    
def plot_encounter():
    """"""
    orbit_dict = pickle.load(open('../data/encounter.pkl', 'rb'))
    orbits = gd.Orbit(pos=orbit_dict['pos'], vel=orbit_dict['vel'], t=orbit_dict['t'])
    
    orbit_gc = orbits[:,0]
    orbit_stream = orbits[:,1:]
    
    stream = orbit_stream[-1].to_coord_frame(gc.GD1)
    stream_init = pickle.load(open('../data/gd1_section_present.pkl', 'rb'))

    #y_init = [stream_init.phi2, stream_init.distance, stream_init.pm_phi1_cosphi2, stream_init.pm_phi2, stream_init.radial_velocity]
    #y = [stream.phi2, stream.distance, stream.pm_phi1_cosphi2, stream.pm_phi2, stream.radial_velocity]

    y_init = [stream_init.phi2, stream_init.pm_phi1_cosphi2, stream_init.pm_phi2, stream_init.radial_velocity]
    y = [stream.phi2, stream.pm_phi1_cosphi2, stream.pm_phi2, stream.radial_velocity]
    ylabels = ['$\phi_2$ [deg]', '$\mu_{\phi_1}$ [mas yr$^{-1}$]', '$\mu_{\phi_2}$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']

    plt.close()
    fig, ax = plt.subplots(4,1, figsize=(12,12), sharex=True)
    
    for i in range(4):
        plt.sca(ax[i])
        plt.plot(stream_init.phi1, y_init[i], 'k-', label='Unperturbed')
        plt.plot(stream.phi1, y[i], 'r-', label='Interaction with NGC 1851')
        
        plt.ylabel(ylabels[i])
    
    plt.xlabel('$\phi_1$ [deg]')
    plt.legend(fontsize='small')
    
    #plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('../plots/stream_encounter_ngc1851.png')


def save_giants():
    """"""
    # H3 data
    rcat = Table(fits.open('/home/ana/data/rcat.fits')[1].data)
    giants = rcat[(rcat['logg']<3.5) & (rcat['SNR']>3) & (rcat['FLAG']==0)]
    
    giants.write('../data/h3_giants.fits', overwrite=True)

def shell_evolution(Tnum=0):
    """"""
    
    # H3 data
    #rcat = Table(fits.open('/home/ana/data/rcat.fits')[1].data)
    #giants = rcat[(rcat['logg']<3.5) & (rcat['SNR']>3) & (rcat['FLAG']==0)]
    giants = Table.read('../data/h3_giants.fits')
    td_flag = ((giants['aFe'] - (0.25 -0.5*(giants['FeH']+0.7)))  > 0) & (giants['aFe']<0.46)

    ind_vod = (giants['eccen_pot1']>0.75) & ((giants['Z_gal'])>10) & ((giants['Z_gal'])<30) & (giants['X_gal']<0) & (giants['Y_gal']<0) & (~td_flag) & (giants['Sgr_FLAG']!=1) & (np.abs(giants['X_gal'])<20) & (np.abs(giants['Y_gal'])<20)
    ind_hac = (giants['eccen_pot1']>0.75) & ((giants['Z_gal'])<-10) & ((giants['Z_gal'])>-30) & (giants['X_gal']>0) & (giants['Y_gal']>0) & (~td_flag) & (giants['Sgr_FLAG']!=1) & (np.abs(giants['X_gal'])<20) & (np.abs(giants['Y_gal'])<20)
    ind_vod = (giants['eccen_pot1']>0.75) & (~td_flag) & (giants['Sgr_FLAG']!=1)
    print(np.sum(ind_vod), np.sum(ind_hac))
    
    #ind_vod = (giants['eccen_pot1']>0.75) & (~td_flag) & (giants['Sgr_FLAG']!=1)
    
    # coordinates
    c = coord.SkyCoord(ra=giants['RA']*u.deg, dec=giants['DEC']*u.deg, distance=giants['dist_adpt']*u.kpc, pm_ra_cosdec=giants['GAIAEDR3_PMRA']*u.mas/u.yr, pm_dec=giants['GAIAEDR3_PMDEC']*u.mas/u.yr, radial_velocity=giants['Vrad']*u.km/u.s, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    ## gravitational potentials
    #mw = gp.MilkyWayPotential()
    ##print(mw.values())
    
    #tot_mass = 5.4e11
    #f = 0.113
    #mw_mass = tot_mass * (1-f)
    #gse_mass = tot_mass * f
    
    #mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    #gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1, c=1, units=galactic, R=R.from_euler('yz', (-35, -35), degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1, c=1, units=galactic, R=R.from_euler('xyz', (120, 60, -30), degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.3, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, 30), degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.4, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-90, -30, -120), degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=13.3, a=1.16, b=1.14, c=1, units=galactic, R=R.from_euler('xyz', (47, 10, 13), degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=13.3, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.4, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (-110, -20, 70), degrees=True))
    #gse = gp.LeeSutoTriaxialNFWPotential(150*u.km/u.s, 16*u.kpc, 1, 0.57, 0.9, units=galactic, R=R.from_euler('XYZ',(-47,-28,-41.57),degrees=True))
    #gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1, b=0.57, c=0.9, units=galactic, R=R.from_euler('XYZ', (-47,-28,-41.57), degrees=True))
    #mw_tilt = mw_part + gse
    mw = ham.potential
    mw_tilt = ham_tilt.potential
    
    # orbit integrations
    T = Tnum*u.Gyr
    dt = 1*u.Myr
    nstep = int((T/dt).decompose())
    
    #orbit_hac = mw.integrate_orbit(w0[ind_hac], dt=dt, n_steps=nstep)
    #orbit_hac_tilt = mw_tilt.integrate_orbit(w0[ind_hac], dt=dt, n_steps=nstep)
    
    orbit_vod = mw.integrate_orbit(w0[ind_vod], dt=dt, n_steps=nstep)
    orbit_vod_tilt = mw_tilt.integrate_orbit(w0[ind_vod], dt=dt, n_steps=nstep)
    
    periods = orbit_vod_tilt.estimate_period()
    #print(np.size(periods), np.sum(ind_vod))
    #print(np.nanpercentile(periods, [16,50,84]))
    
    
    #orbits = [orbit_hac, orbit_hac_tilt]
    orbits = [orbit_vod, orbit_vod_tilt]
    
    ind = orbits[0].spherical.pos.distance[-1]<50*u.kpc
    cxy = np.corrcoef(orbits[0].pos.x[-1][ind].value, orbits[0].pos.y[-1][ind].value)[0,1]
    cxz = np.corrcoef(orbits[0].pos.x[-1][ind].value, orbits[0].pos.z[-1][ind].value)[0,1]
    cyz = np.corrcoef(orbits[0].pos.y[-1][ind].value, orbits[0].pos.z[-1][ind].value)[0,1]

    #m = np.array([orbits[0].pos.x[-1][ind], orbits[0].pos.y[-1][ind], orbits[0].pos.z[-1][ind]])
    #print(np.cov(m))
    
    ind = orbits[1].spherical.pos.distance[-1]<50*u.kpc
    cxy_tilt = np.corrcoef(orbits[1].pos.x[-1][ind].value, orbits[1].pos.y[-1][ind].value)[0,1]
    cxz_tilt = np.corrcoef(orbits[1].pos.x[-1][ind].value, orbits[1].pos.z[-1][ind].value)[0,1]
    cyz_tilt = np.corrcoef(orbits[1].pos.y[-1][ind].value, orbits[1].pos.z[-1][ind].value)[0,1]

    label_1 = [cxy, cxy_tilt]
    label_2 = [cxz, cxz_tilt]
    label_3 = [cyz, cyz_tilt]
    label_4 = ['Axisymmetric', 'Tilted']
    #m = np.array([orbits[1].pos.x[-1][ind], orbits[1].pos.y[-1][ind], orbits[1].pos.z[-1][ind]])
    #print(np.cov(m))
    
    plt.close()
    fig, ax = plt.subplots(2,3, figsize=(12,7.5), sharex='col', sharey='col')
    
    for i in range(2):
        plt.sca(ax[i][0])
        #plt.plot(orbits[i].pos.x, orbits[i].pos.y, 'k-', lw=0.5, alpha=0.5)
        plt.plot(orbits[i].pos.x[-1], orbits[i].pos.y[-1], 'k.', mew=0, alpha=0.5)
        plt.ylabel('Y [kpc]')
        plt.text(0.1, 0.9, '{:s}'.format(label_4[i]), fontsize='small', transform=plt.gca().transAxes, va='top', ha='left')
        plt.text(0.9, 0.9, 'r = {:.3}'.format(label_1[i]), fontsize='small', transform=plt.gca().transAxes, va='top', ha='right')
        
        
        plt.sca(ax[i][1])
        #plt.plot(orbits[i].pos.x, orbits[i].pos.z, 'k-', lw=0.5, alpha=0.5)
        plt.plot(orbits[i].pos.x[-1], orbits[i].pos.z[-1], 'k.', mew=0, alpha=0.5)
        plt.ylabel('Z [kpc]')
        plt.text(0.9, 0.9, 'r = {:.3}'.format(label_2[i]), fontsize='small', transform=plt.gca().transAxes, va='top', ha='right')
        
        plt.sca(ax[i][2])
        #plt.plot(orbits[i].pos.y, orbits[i].pos.z, 'k-', lw=0.5, alpha=0.5)
        plt.plot(orbits[i].pos.y[-1], orbits[i].pos.z[-1], 'k.', mew=0, alpha=0.5)
        plt.ylabel('Z [kpc]')
        plt.text(0.9, 0.9, 'r = {:.3}'.format(label_3[i]), fontsize='small', transform=plt.gca().transAxes, va='top', ha='right')
    
    xlabels = ['X', 'X', 'Y']
    for i in range(3):
        plt.sca(ax[0][i])
        plt.xlim(-50,50)
        plt.ylim(-50,50)
        plt.gca().set_aspect('equal')

        plt.sca(ax[1][i])
        #plt.xlim(-50,50)
        #plt.ylim(-50,50)
        plt.gca().set_aspect('equal')
        plt.xlabel('{:s} [kpc]'.format(xlabels[i]))
    
    
    plt.sca(ax[0][1])
    plt.title('{:g}'.format(T), fontsize='medium')
    
    plt.tight_layout()
    #plt.savefig('../plots/vod_{:.1f}.png'.format(T.to(u.Gyr).value))

def tilt_grid():
    """"""
    giants = Table.read('../data/h3_giants.fits')
    td_flag = ((giants['aFe'] - (0.25 -0.5*(giants['FeH']+0.7)))  > 0) & (giants['aFe']<0.46)
    ind_vod = (giants['eccen_pot1']>0.75) & ((giants['Z_gal'])>10) & ((giants['Z_gal'])<30) & (giants['X_gal']<0) & (giants['Y_gal']<0) & (~td_flag) & (giants['Sgr_FLAG']!=1) & (np.abs(giants['X_gal'])<20) & (np.abs(giants['Y_gal'])<20)
    giants = giants[ind_vod]
    
    # coordinates
    c = coord.SkyCoord(ra=giants['RA']*u.deg, dec=giants['DEC']*u.deg, distance=giants['dist_adpt']*u.kpc, pm_ra_cosdec=giants['GAIAEDR3_PMRA']*u.mas/u.yr, pm_dec=giants['GAIAEDR3_PMDEC']*u.mas/u.yr, radial_velocity=giants['Vrad']*u.km/u.s, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)

    # potential
    tot_mass = 5.4e11
    f = 0.17
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    
    # orbit integrations
    T = 2*u.Gyr
    dt = 1*u.Myr
    nstep = int((T/dt).decompose())
    
    arr_psi = np.arange(-180, 180, 10)
    arr_theta = np.arange(-90, 90, 10)
    arr_phi = np.arange(-180, 180, 10)
    
    for psi in arr_psi:
        for theta in arr_theta:
            for phi in arr_phi:
                gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.4, b=1.1, c=1, units=galactic, R=R.from_euler('xyz', (psi, theta, phi), degrees=True))
                mw_tilt = mw_part + gse
                
                w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
                orbit = mw_tilt.integrate_orbit(w0, dt=dt, n_steps=nstep)
                
                ind = orbit.spherical.pos.distance[-1]<50*u.kpc
                cxy = np.corrcoef(orbit.pos.x[-1][ind].value, orbit.pos.y[-1][ind].value)[0,1]
                cxz = np.corrcoef(orbit.pos.x[-1][ind].value, orbit.pos.z[-1][ind].value)[0,1]
                cyz = np.corrcoef(orbit.pos.y[-1][ind].value, orbit.pos.z[-1][ind].value)[0,1]
                
                if (cxy>0.2) & (cxz<-0.2) & (cyz<-0.2):
                    print(psi, theta, phi, cxy, cxz, cyz)


def impact_orbits():
    """"""
    
    # Read in gaps' 6D position
    c_g40 = pickle.load(open('../data/gap40_location.pkl', 'rb'))
    w0_g40 = gd.PhaseSpacePosition(c_g40.transform_to(gc_frame).cartesian)

    c_g20 = pickle.load(open('../data/gap20_location.pkl', 'rb'))
    w0_g20 = gd.PhaseSpacePosition(c_g20.transform_to(gc_frame).cartesian)
    
    
    # Read in clusters
    tall = Table.read('../data/gc_vasiliev.fits')
    name = 'NGC 1851'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n1851 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n1851 = gd.PhaseSpacePosition(c_n1851.transform_to(gc_frame).cartesian)
    
    name = 'NGC 4590'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n4590 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n4590 = gd.PhaseSpacePosition(c_n4590.transform_to(gc_frame).cartesian)
    
    
    # potential
    tot_mass = 5.4e11
    f = 0.113
    rs = 13.3
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    mw_tilt = mw_part + gse
    
    
    # integrate orbits
    Tback = 1*u.Gyr
    dt = -0.1*u.Myr
    nstep = np.abs(np.int((Tback/dt).decompose()))
    
    
    orbit_g20 = mw_tilt.integrate_orbit(w0_g20, dt=dt, n_steps=nstep)
    orbit_g40 = mw_tilt.integrate_orbit(w0_g40, dt=dt, n_steps=nstep)
    orbit_n1851 = mw_tilt.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    orbit_n4590 = mw_tilt.integrate_orbit(w0_n4590, dt=dt, n_steps=nstep)

    orbits = [orbit_g20, orbit_g40, orbit_n1851, orbit_n4590]
    colors = ['0.2', '0.5', 'r', 'b']
    
    pos_g20 = np.array([orbit_g20.pos.x, orbit_g20.pos.y, orbit_g20.pos.z])
    pos_g40 = np.array([orbit_g40.pos.x, orbit_g40.pos.y, orbit_g40.pos.z])
    pos_n1851 = np.array([orbit_n1851.pos.x, orbit_n1851.pos.y, orbit_n1851.pos.z])
    pos_n4590 = np.array([orbit_n4590.pos.x, orbit_n4590.pos.y, orbit_n4590.pos.z])
    
    dist_g20n1851 = np.linalg.norm(pos_g20 - pos_n1851, axis=0)
    dist_g20n4590 = np.linalg.norm(pos_g20 - pos_n4590, axis=0)
    dist_g40n1851 = np.linalg.norm(pos_g40 - pos_n1851, axis=0)
    dist_g40n4590 = np.linalg.norm(pos_g40 - pos_n4590, axis=0)

    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.plot(orbits[0].t, dist_g20n1851, '-', color='tab:red', label='Gap 20, NGC 1851')
    plt.plot(orbits[0].t, dist_g20n4590, '-', color='tab:orange', label='Gap 20, NGC 4590')
    plt.plot(orbits[0].t, dist_g40n1851, '-', color='tab:blue', label='Gap 40, NGC 1851')
    plt.plot(orbits[0].t, dist_g40n4590, '-', color='tab:green', label='Gap 40, NGC 4590')
    
    plt.legend()
    plt.gca().set_yscale('log')
    plt.tight_layout()


def impact_projections():
    """"""
    # Read in gaps' 6D position
    c_g40 = pickle.load(open('../data/gap40_location.pkl', 'rb'))
    w0_g40 = gd.PhaseSpacePosition(c_g40.transform_to(gc_frame).cartesian)

    c_g20 = pickle.load(open('../data/gap20_location.pkl', 'rb'))
    w0_g20 = gd.PhaseSpacePosition(c_g20.transform_to(gc_frame).cartesian)
    
    
    # Read in clusters
    tall = Table.read('../data/gc_vasiliev.fits')
    name = 'NGC 1851'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n1851 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n1851 = gd.PhaseSpacePosition(c_n1851.transform_to(gc_frame).cartesian)
    
    name = 'NGC 4590'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n4590 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n4590 = gd.PhaseSpacePosition(c_n4590.transform_to(gc_frame).cartesian)
    
    
    # potential
    tot_mass = 5.4e11
    f = 0.113
    rs = 13.3
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    mw_tilt = mw_part + gse
    
    
    # integrate orbits
    Tback = 0.5*u.Gyr
    dt = -0.1*u.Myr
    nstep = np.abs(np.int((Tback/dt).decompose()))
    
    orbit_g20 = mw_tilt.integrate_orbit(w0_g20, dt=dt, n_steps=nstep)
    orbit_g40 = mw_tilt.integrate_orbit(w0_g40, dt=dt, n_steps=nstep)
    orbit_n1851 = mw_tilt.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    orbit_n4590 = mw_tilt.integrate_orbit(w0_n4590, dt=dt, n_steps=nstep)

    pos_g20 = np.array([orbit_g20.pos.x, orbit_g20.pos.y, orbit_g20.pos.z])
    pos_g40 = np.array([orbit_g40.pos.x, orbit_g40.pos.y, orbit_g40.pos.z])
    pos_n1851 = np.array([orbit_n1851.pos.x, orbit_n1851.pos.y, orbit_n1851.pos.z])
    pos_n4590 = np.array([orbit_n4590.pos.x, orbit_n4590.pos.y, orbit_n4590.pos.z])
    
    dist_g20n1851 = np.linalg.norm(pos_g20 - pos_n1851, axis=0)
    dist_g20n4590 = np.linalg.norm(pos_g20 - pos_n4590, axis=0)
    dist_g40n1851 = np.linalg.norm(pos_g40 - pos_n1851, axis=0)
    dist_g40n4590 = np.linalg.norm(pos_g40 - pos_n4590, axis=0)

    ind_min = [np.argmin(dist_g20n4590), np.argmin(dist_g40n1851), np.argmin(dist_g40n1851), np.argmin(dist_g20n4590)]

    orbits = [orbit_g20, orbit_g40, orbit_n1851, orbit_n4590]
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(18,6))
    
    for e, orbit in enumerate(orbits):
        plt.sca(ax[0])
        plt.plot(orbit.pos.x[0], orbit.pos.y[0], 'o', color=colors[e])
        plt.plot(orbit.pos.x[ind_min[e]], orbit.pos.y[ind_min[e]], '^', color=colors[e])
        plt.plot(orbit.pos.x, orbit.pos.y, '-', color=colors[e])
        
        plt.sca(ax[1])
        plt.plot(orbit.pos.x[0], orbit.pos.z[0], 'o', color=colors[e])
        plt.plot(orbit.pos.x[ind_min[e]], orbit.pos.z[ind_min[e]], '^', color=colors[e])
        plt.plot(orbit.pos.x, orbit.pos.z, '-', color=colors[e])
        
        plt.sca(ax[2])
        plt.plot(orbit.pos.y[0], orbit.pos.z[0], 'o', color=colors[e])
        plt.plot(orbit.pos.y[ind_min[e]], orbit.pos.z[ind_min[e]], '^', color=colors[e])
        plt.plot(orbit.pos.y, orbit.pos.z, '-', color=colors[e])
    
    plt.tight_layout()


def impact_sky():
    """"""
    
    # H3 data
    tg = Table.read('../data/h3_giants.fits')
    #ind = tg['dist_adpt']>5
    #tg = tg[ind]
    c_nr = coord.SkyCoord(ra=tg['RA']*u.deg, dec=tg['DEC']*u.deg, distance=tg['dist_adpt']*u.kpc, pm_ra_cosdec=tg['GAIAEDR3_PMRA']*u.mas/u.yr, pm_dec=tg['GAIAEDR3_PMDEC']*u.mas/u.yr, radial_velocity=tg['Vrad']*u.km/u.s)
    c = gc.reflex_correct(c_nr, galactocentric_frame=gc_frame)
    #c = c_nr
    
    # Read in gaps' 6D position
    c_g40 = pickle.load(open('../data/gap40_location.pkl', 'rb'))
    w0_g40 = gd.PhaseSpacePosition(c_g40.transform_to(gc_frame).cartesian)

    c_g20 = pickle.load(open('../data/gap20_location.pkl', 'rb'))
    w0_g20 = gd.PhaseSpacePosition(c_g20.transform_to(gc_frame).cartesian)
    
    
    # Read in clusters
    tall = Table.read('../data/gc_vasiliev.fits')
    name = 'NGC 1851'
    name = 'NGC 6864'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n1851 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n1851 = gd.PhaseSpacePosition(c_n1851.transform_to(gc_frame).cartesian)
    
    name = 'NGC 4590'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n4590 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n4590 = gd.PhaseSpacePosition(c_n4590.transform_to(gc_frame).cartesian)
    
    
    # potential
    tot_mass = 5.4e11
    f = 0.113
    rs = 13.3
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    mw_tilt = mw_part + gse
    
    mw = gp.MilkyWayPotential()
    
    
    # integrate orbits
    Tback = 0.6*u.Gyr
    dt = -0.1*u.Myr
    nstep = np.abs(np.int((Tback/dt).decompose()))
    
    orbit_g40 = mw_tilt.integrate_orbit(w0_g40, dt=dt, n_steps=nstep)
    orbit_n1851 = mw_tilt.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    
    ## reverse the orbit
    #w1_n1851 = orbit_n1851[-1]
    #orbit_n1851 = mw_tilt.integrate_orbit(w1_n1851, dt=-dt, n_steps=nstep)

    pos_g40 = np.array([orbit_g40.pos.x, orbit_g40.pos.y, orbit_g40.pos.z])
    pos_n1851 = np.array([orbit_n1851.pos.x, orbit_n1851.pos.y, orbit_n1851.pos.z])
    
    dist_g40n1851 = np.linalg.norm(pos_g40 - pos_n1851, axis=0)
    ind = np.argmin(dist_g40n1851)
    
    sky_g40 = orbit_g40.to_coord_frame(coord.ICRS)
    sky_n1851 = orbit_n1851.to_coord_frame(coord.ICRS)

    forbit_g40 = mw.integrate_orbit(w0_g40, dt=dt, n_steps=nstep)
    forbit_n1851 = mw.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)

    fpos_g40 = np.array([forbit_g40.pos.x, forbit_g40.pos.y, forbit_g40.pos.z])
    fpos_n1851 = np.array([forbit_n1851.pos.x, forbit_n1851.pos.y, forbit_n1851.pos.z])
    
    fdist_g40n1851 = np.linalg.norm(fpos_g40 - fpos_n1851, axis=0)
    find = np.argmin(fdist_g40n1851)
    print(np.min(dist_g40n1851))
    print(np.min(fdist_g40n1851))
    
    fsky_g40 = forbit_g40.to_coord_frame(coord.ICRS)
    fsky_n1851 = forbit_n1851.to_coord_frame(coord.ICRS)
    
    fsky = gc.reflex_correct(fsky_n1851, galactocentric_frame=gc_frame)[::10]
    sky = gc.reflex_correct(sky_n1851, galactocentric_frame=gc_frame)[::10]
    sky = sky_n1851[::100]
    pmtot_sky = np.sqrt(sky.pm_ra_cosdec.value**2 + sky.pm_dec.value**2)
    
    plt.close()
    plt.figure(figsize=(12,12))
    
    plt.plot(sky_g40.ra, sky_g40.dec, '-', color='tab:blue', label='Gap -40 tilt')
    plt.plot(sky_g40.ra[ind], sky_g40.dec[ind], 'o', color='tab:blue', label='')
    plt.plot(sky_n1851.ra, sky_n1851.dec, '-', color='tab:orange', label='NGC1851 tilt')
    plt.plot(sky_n1851.ra[ind], sky_n1851.dec[ind], 'o', color='tab:orange', label='')
    
    plt.plot(fsky_g40.ra, fsky_g40.dec, ':', color='tab:blue', label='Gap -40 fiducial')
    plt.plot(fsky_g40.ra[find], fsky_g40.dec[find], 'o', color='tab:blue', label='')
    plt.plot(fsky_n1851.ra, fsky_n1851.dec, ':', color='tab:orange', label='NGC1851 fiducial')
    plt.plot(fsky_n1851.ra[find], fsky_n1851.dec[find], 'o', color='tab:orange', label='')
    plt.quiver(sky.ra.value, sky.dec.value, sky.pm_ra_cosdec.value/pmtot_sky, sky.pm_dec.value/pmtot_sky, color='tab:orange', zorder=1, alpha=0.2)
    
    plt.plot(tg['RA'], tg['DEC'], 'k.', zorder=0, label='H3')
    
    ind_chem = (tg['FeH']>-1.35) & (tg['FeH']<-1.19)
    ind_chem = (tg['FeH']>-1.23) & (tg['FeH']<-1.13)
    ind_chem = (tg['FeH']>-2.6) & (tg['FeH']<-2)
    ind_elz = (np.abs(tg['E_tot_pot1']+0.115)<0.01) & (np.abs(tg['Lz'] - 0.16)<0.1)
    pmtot = np.sqrt(c.pm_ra_cosdec.value**2 + c.pm_dec.value**2)

    #plt.plot(tg['RA'][ind_chem & ind_elz], tg['DEC'][ind_chem & ind_elz], 'ro')
    #plt.quiver(tg['RA'][ind_chem], tg['DEC'][ind_chem], tg['GAIAEDR3_PMRA'][ind_chem], tg['GAIAEDR3_PMDEC'][ind_chem], color='r', zorder=1)
    plt.quiver(c.ra.value[ind_chem], c.dec.value[ind_chem], (c.pm_ra_cosdec.value/pmtot)[ind_chem], (c.pm_dec.value/pmtot)[ind_chem], color='r', zorder=1, scale=40)

    #plt.quiver(c.ra.value[ind_chem & ind_elz], c.dec.value[ind_chem & ind_elz], c.pm_ra_cosdec.value[ind_chem & ind_elz], c.pm_dec.value[ind_chem & ind_elz], color='r', zorder=1)
    print(nstep, np.sum(ind_chem))
    
    # closest to the orbit
    orbit_arr = np.array([fsky.ra.value, fsky.dec.value]).T
    h3_arr = np.array([c.ra.value[ind_chem], c.dec.value[ind_chem]]).T
    
    dist = cdist(orbit_arr, h3_arr) * u.deg
    min_dist = np.min(dist, axis=0)
    print(np.shape(min_dist), np.shape(h3_arr))
    ind_close = min_dist<1.5*u.deg
    print(np.sum(ind_close))
    
    torbit = tg[ind_chem][ind_close]
    #torbit.write('../data/h3_n1851_fiducial.fits', overwrite=True)
    torbit.write('../data/h3_n6864_fiducial.fits', overwrite=True)
    
    plt.legend(fontsize='small', handlelength=0.5)
    plt.xlabel('R.A. [deg]')
    plt.ylabel('Dec [deg]')
    
    #plt.xlim(125, 250)
    #plt.ylim(-25, 75)
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('../plots/impact_sky_orbits.png')

def orbit_distance(tilt=True):
    """"""
    # Read in clusters
    tall = Table.read('../data/gc_vasiliev.fits')
    name = 'NGC 1851'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n1851 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n1851 = gd.PhaseSpacePosition(c_n1851.transform_to(gc_frame).cartesian)
    
    # potential
    tot_mass = 5.4e11
    f = 0.113
    rs = 13.3
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    mw_tilt = mw_part + gse
    
    mw = gp.MilkyWayPotential()
    
    # integrate orbits
    Tback = 0.6*u.Gyr
    dt = -0.1*u.Myr
    nstep = np.abs(np.int((Tback/dt).decompose()))
    
    if tilt:
        orbit_n1851 = mw_tilt.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    else:
        orbit_n1851 = mw.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    
    sky_n1851 = orbit_n1851.to_coord_frame(coord.ICRS)[3100:]

    forbit_n1851 = mw.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    fsky_n1851 = forbit_n1851.to_coord_frame(coord.ICRS)
    
    # h3
    if tilt:
        tg = Table.read('../data/h3_n1851.fits')
    else:
        tg = Table.read('../data/h3_n1851_fiducial.fits')
    ind = tg['dist_adpt']<20
    tg = tg[ind]
    
    sky = [sky_n1851.dec, sky_n1851.dec, sky_n1851.distance, sky_n1851.pm_ra_cosdec, sky_n1851.pm_dec, sky_n1851.radial_velocity]
    h3 = [tg['RA'], tg['DEC'], tg['dist_adpt'], tg['GAIAEDR3_PMRA'], tg['GAIAEDR3_PMDEC'], tg['Vrad']]
    err = [tg['GAIAEDR3_RA_ERROR'], tg['GAIAEDR3_DEC_ERROR'], tg['dist_adpt_err'], tg['GAIAEDR3_PMRA_ERROR'], tg['GAIAEDR3_PMDEC_ERROR'], tg['Vrad_err']]
    print(np.percentile(tg['aFe'], [16,50,84]))

    tg_ = Table.read('../data/h3_n1851_fiducial.fits')
    ind = tg_['dist_adpt']<20
    tg_ = tg_[ind]
    h3_ = [tg_['DEC'], tg_['dist_adpt'], tg_['GAIAEDR3_PMRA'], tg_['GAIAEDR3_PMDEC'], tg_['Vrad']]
    
    wangle = 0*u.deg
    sky_pmtot = np.sqrt(sky_n1851.pm_ra_cosdec**2 + sky_n1851.pm_dec**2)
    h3_pmtot = np.sqrt(tg['GAIAEDR3_PMRA']**2 + tg['GAIAEDR3_PMDEC']**2)
    h3__pmtot = np.sqrt(tg_['GAIAEDR3_PMRA']**2 + tg_['GAIAEDR3_PMDEC']**2)
    
    #sky = [sky_n1851.dec, sky_n1851.dec, sky_n1851.distance, sky_pmtot, coord.Angle(np.arctan2(sky_n1851.pm_ra_cosdec,sky_n1851.pm_dec)).wrap_at(wangle), sky_n1851.radial_velocity]
    #h3 = [tg['RA'], tg['DEC'], tg['dist_adpt'], h3_pmtot, coord.Angle(np.arctan2(tg['GAIAEDR3_PMRA'],tg['GAIAEDR3_PMDEC'])*u.rad).wrap_at(wangle), tg['Vrad']]
    #h3_ = [tg_['RA'], tg_['DEC'], tg_['dist_adpt'], h3__pmtot, coord.Angle(np.arctan2(tg_['GAIAEDR3_PMRA'],tg_['GAIAEDR3_PMDEC'])*u.rad).wrap_at(wangle), tg_['Vrad']]
    
    labels = ['Dec', 'dist', '$\mu_\\alpha$', '$\mu_\delta$', '$V_r$']
    
    plt.close()
    fig, ax = plt.subplots(5,1,figsize=(12,10), sharex=True)
    
    for i in range(5):
        plt.sca(ax[i])
        plt.scatter(sky_n1851.ra, sky[i+1], c=sky[0], cmap='magma', s=5)
        plt.scatter(tg['RA'], h3[i+1], c=h3[0], zorder=1)
        plt.errorbar(tg['RA'], h3[i+1], yerr=err[i+1], fmt='none', color='k', zorder=0)
        #plt.plot(tg_['RA'], h3_[i], 'k.')
        #print(err[i+1])
        
        plt.ylabel(labels[i])
        
    #plt.sca(ax[3])
    #plt.ylim()
    plt.xlabel('R.A.')
    
    plt.tight_layout(h_pad=0)
    plt.savefig('../plots/n1851_stream_search_tilt.{:d}.png'.format(tilt))

def elz():
    """"""
    
    tall = Table.read('../data/gc_vasiliev.fits')
    name = 'NGC 1851'
    ind = [True if name in name_ else False for name_ in tall['Name']]
    t = tall[ind]
    c_n1851 = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['Dist'], pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_n1851 = gd.PhaseSpacePosition(c_n1851.transform_to(gc_frame).cartesian)
    
    # potential
    tot_mass = 5.4e11
    f = 0.113
    rs = 13.3
    mw_mass = tot_mass * (1-f)
    gse_mass = tot_mass * f
    
    mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
    gse = gp.NFWPotential(m=gse_mass, r_s=rs, a=1.08, b=1.08, c=1, units=galactic, R=R.from_euler('xyz', (75, 0, 0), degrees=True))
    mw_tilt = mw_part + gse
    
    mw = gp.MilkyWayPotential()
    
    # integrate orbits
    Tback = 0.5*u.Gyr
    dt = -0.1*u.Myr
    nstep = np.abs(np.int((Tback/dt).decompose()))
    
    orbit_n1851 = mw_tilt.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    forbit_n1851 = mw.integrate_orbit(w0_n1851, dt=dt, n_steps=nstep)
    
    print(orbit_n1851.energy()[0], orbit_n1851.angular_momentum()[:,0])
    print(forbit_n1851.energy()[0], forbit_n1851.angular_momentum()[:,0])
    
    fetot = forbit_n1851.energy()[0]
    flz = forbit_n1851.angular_momentum()[2,0]
    etot = orbit_n1851.energy()[0]
    lz = orbit_n1851.angular_momentum()[2,0]
    
    tg = Table.read('../data/h3_giants.fits')
    tg['E_tot_pot1'] *= (u.km/u.s)**2
    tg['E_tot_pot1'] = tg['E_tot_pot1'].to(u.kpc**2 * u.Myr**-2)
    tg['Lz'] *= (u.kpc*u.km/u.s)
    tg['Lz'] = tg['Lz'].to(u.kpc**2 * u.Myr**-1)
    
    ind_chem = (tg['FeH']>-1.3) & (tg['FeH']<-1.24)
    ind_elz = (np.abs(tg['E_tot_pot1']+0.115)<0.007) & (np.abs(tg['Lz'] - 0.16)<0.1)
    
    plt.close()
    plt.figure(figsize=(8,8))
    
    plt.plot(tg['Lz'], tg['E_tot_pot1'], 'k.', ms=1)
    #plt.plot(tg['Lz'][ind_chem], tg['E_tot_pot1'][ind_chem], 'k.', ms=1)
    #plt.plot(tg['Lz'][ind_chem & ind_elz], tg['E_tot_pot1'][ind_chem & ind_elz], 'ro')
    #plt.plot(tg['Lz'][ind_chem], tg['E_tot_pot1'][ind_chem], 'ro')
    
    #plt.plot(lz, etot, 'o', color='none', mec='cyan', ms=20, mew=2)
    #plt.plot(flz, fetot, 'o', color='none', mec='cyan', ms=15, mew=2)
    
    #plt.xlim(-2,2)
    #plt.ylim(-0.14, -0.08)
    
    plt.tight_layout()
