import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table, Column #, QTable, hstack, vstack
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
from scipy.interpolate import InterpolatedUnivariateSpline

import pickle
import emcee
from multiprocessing import Pool
import corner
import time


ham_fiducial = gp.Hamiltonian(gp.MilkyWayPotential())
ham_bovy = gp.Hamiltonian(gp.BovyMWPotential2014())
ham_heavy = gp.Hamiltonian(gp.MilkyWayPotential(nucleus=dict(m=0), halo=dict(c=0.95, m=7e11), bulge=dict(m=4e9), disk=dict(m=5.5e10)))

tot_mass = 5.4e11
f_gse = 0.171
mw_mass = tot_mass * (1 - f_gse)
gse_mass = tot_mass * f_gse

mw_part = gp.MilkyWayPotential(halo={'m': mw_mass})
gse = gp.NFWPotential(m=gse_mass, r_s=16, a=1.16, b=1.14, c=1, units=galactic, R=R.from_euler('xyz', (47, 10, 13), degrees=True))
ham_tilt = gp.Hamiltonian(mw_part + gse)

gse_alt = gp.NFWPotential(m=gse_mass, r_s=16, a=1.16, b=1.14, c=1, units=galactic, R=R.from_euler('xyz', (47, -6, 30), degrees=True))
ham_tilt_alt = gp.Hamiltonian(mw_part + gse_alt)

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()

class GeneralStream():
    def __init__(self, name, label='', wangle=360*u.deg, ra0=np.nan*u.deg, tstream=50*u.Myr, dt=-0.5*u.Myr, vnorm=1., pmnorm=1., minra=True, dra=0.5*u.deg, ham=ham_fiducial, gc_frame=gc_frame, save_ext=''):
        self.name = name
        if len(save_ext):
            self.savename = '{:s}_{:s}'.format(self.name, save_ext)
        else:
            self.savename = self.name
        if len(label):
            self.label = label
        else:
            self.label = self.name
        
        self.data = pickle.load(open('../data/streams/data_{:s}.pkl'.format(self.name), 'rb'))
        
        self.wangle = wangle
        if ~np.isfinite(ra0.value):
            self.ra0 = self.get_ra0(minra=min, dra=dra)
        else:
            self.ra0 = ra0
        
        self.dt = dt
        self.tstream = tstream
        self.nstep = int((self.tstream/np.abs(self.dt)).decompose())
        
        self.ham = ham
        self.gc_frame = gc_frame
        
        self.rm_dataunits()
            
    def get_ra0(self, minra=True, dra=0.5*u.deg):
        """Select min/max RA as the orbital fiducial point"""
        
        if minra:
            ra0 = np.min(data['phi2'][0]) - dra
        else:
            ra0 = np.max(data['phi2'][1]) + dra
        
        return ra0
    
    def rm_dataunits(self):
        """"""
        self.data_nounits = dict()
        for k in self.data.keys():
            self.data_nounits[k] = [x.value for x in self.data[k]]
    
    def orbit_minimize(self, p0=[], save=True):
        """Find best-fitting orbit by maximizing log likelihood"""
        
        if len(p0)==0:
            p0 = self.p0
        
        #self.rm_dataunits()
        p0_input = [x_.value for x_ in p0]
        
        res = minimize(lambda *x: -ln_likelihood_icrs(*x), x0=p0_input, args=(self.ra0, self.data_nounits, self.nstep, self.dt, self.wangle, self.ham, self.gc_frame, self.fra))
        self.pbest = res.x
        
        if save:
            pickle.dump(res, open('../data/fits/minimization_{:s}.pkl'.format(self.savename), 'wb'))
        
        return res
    
    def orbital_properties(self, pbest=[], t=5*u.Gyr, save=True):
        """"""
        if len(pbest)==0:
            pbest = self.pbest
        
        dec, d, pmra, pmdec, vr = pbest
        
        c = coord.ICRS(ra=self.ra0*u.deg, dec=dec*u.deg, distance=d*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
        w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
        
        n_long = int((t/np.abs(self.dt)).decompose())
        long_orbit = self.ham.integrate_orbit(w0, dt=self.dt, n_steps=n_long)
        
        if save:
            name = np.array([self.name])
            rperi = long_orbit.pericenter()
            rperi = np.array([rperi.value]) * rperi.unit
            rapo = long_orbit.apocenter()
            rapo = np.array([rapo.value]) * rapo.unit
            ecc = np.array([long_orbit.eccentricity()])
            vcirc = self.ham.potential.circular_velocity(np.array([long_orbit.apocenter().to(u.kpc).value, 0, 0]))
            vcirc = np.array([vcirc.value]) * vcirc.unit
            
            tout = Table([name, rperi, rapo, ecc, vcirc], names=('name', 'rperi', 'rapo', 'ecc', 'vcirc'))
            tout.write('../data/fits/minimization_orbit_{:s}.fits'.format(self.savename), overwrite=True)
        
        return long_orbit

class Stream(GeneralStream):
    def __init__(self, name, dt=-0.5*u.Myr, ham=ham_fiducial, gc_frame=gc_frame, save_ext=''):
        prop = get_properties(name)
        self._prop = prop
        
        self.name = name
        if len(save_ext):
            self.savename = '{:s}_{:s}'.format(self.name, save_ext)
        else:
            self.savename = self.name
        self.label = prop['label']
        self.data = pickle.load(open('../data/streams/data_{:s}.pkl'.format(self.name), 'rb'))
        
        self.wangle = prop['wangle']
        self.fra = prop['fra']
        self.ra0 = prop['ra0'].value
        self.p0 = [prop[x] for x in ['dec0', 'd0', 'pmra0', 'pmdec0', 'vr0']]
        
        self.dt = dt
        self.tstream = prop['tstream']
        self.nstep = int((self.tstream/np.abs(self.dt)).decompose())
        
        self.ham = ham
        self.gc_frame = gc_frame
        
        self.rm_dataunits()

def ln_likelihood_icrs(p, ra_0, data, n_steps, dt, wangle, ham, gc_frame, fra):
    # initial conditions at ra_0
    dec, d, pmra, pmdec, vr = p
    
    if (d<0) | (np.abs(vr)>500) | (dec<-90) | (dec>90):
        return -np.inf
    
    wdeg = wangle.to(u.deg).value
    
    c = coord.ICRS(ra=ra_0*u.deg, dec=dec*u.deg, distance=d*u.kpc, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, radial_velocity=vr*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    model_stream = orbit.to_coord_frame(coord.ICRS, galactocentric_frame=gc_frame)
    
    model_ra = model_stream.ra.wrap_at(wangle).degree
    if model_ra[-1] < wdeg - 360:
        return -np.inf
    model_dec = model_stream.dec.degree
    
    if fra:
        model_x = model_ra
        model_y = model_dec
        indx = 0
        bbox = [wdeg - 360, wdeg]
    else:
        model_x = model_dec
        model_y = model_ra
        indx = -1
        bbox = [-90, 90]
        
        # switch data order
        data['dec'][1] = data['dec'][0]
    
    model_dist = model_stream.distance.to(u.kpc).value
    model_pmra = model_stream.pm_ra_cosdec.to(u.mas/u.yr).value
    model_pmdec = model_stream.pm_dec.to(u.mas/u.yr).value
    model_vr = model_stream.radial_velocity.to(u.km/u.s).value

    ix = np.argsort(model_x)
    model_x = model_x[ix]
    
    # define interpolating functions
    order = 3
    
    interp = {}
    interp['dec'] = InterpolatedUnivariateSpline(model_x, model_y[ix], k=order, bbox=bbox)
    interp['dist'] = InterpolatedUnivariateSpline(model_x, model_dist[ix], k=order, bbox=bbox)
    interp['pmra'] = InterpolatedUnivariateSpline(model_x, model_pmra[ix], k=order, bbox=bbox)
    interp['pmdec'] = InterpolatedUnivariateSpline(model_x, model_pmdec[ix], k=order, bbox=bbox)
    interp['vr'] = InterpolatedUnivariateSpline(model_x, model_vr[ix], k=order, bbox=bbox)
    
    # model smoothing
    isigma = {}
    isigma['dec'] = 0.01 # deg
    isigma['dist'] = 0.1 # kpc
    isigma['pmra'] = 0. # mas/yr
    isigma['pmdec'] = 0. # mas/yr
    isigma['vr'] = 1 # km/s
    
    chi2 = 0
    keys = data.keys()
    for k in keys:
        sigma = np.sqrt(isigma[k]**2 + data[k][2]**2)
        chi2 += np.sum(-(interp[k](data[k][indx]) - data[k][1])**2 / sigma**2 - 2*np.log(sigma))
    
    return chi2


def get_properties(name):
    """Return initial positions"""
    
    props = {}

    props['ophiuchus'] = dict(label='Ophiuchus', wangle=360*u.deg, ra0=240.5*u.deg, dec0=-7.3*u.deg, d0=10*u.kpc, pmra0=-4*u.mas/u.yr, pmdec0=-4.5*u.mas/u.yr, vr0=270*u.km/u.s, tstream=13*u.Myr, fra=True, provenance=[7,7,7])
    
    props['gd1'] = dict(label='GD-1', wangle=360*u.deg, ra0=123*u.deg, dec0=-10*u.deg, d0=9*u.kpc, pmra0=-2*u.mas/u.yr, pmdec0=-7*u.mas/u.yr, vr0=300*u.km/u.s, tstream=110*u.Myr, fra=True, provenance=[5,5,6])
    
    props['svol'] = dict(label='Sv\\"{o}l', wangle=360*u.deg, ra0=250*u.deg, dec0=25*u.deg, d0=8*u.kpc, pmra0=3.5*u.mas/u.yr, pmdec0=-6*u.mas/u.yr, vr0=-150*u.km/u.s, tstream=30*u.Myr, fra=True, provenance=[4,2,np.nan])
    
    props['leiptr'] = dict(label='Leiptr', wangle=360*u.deg, ra0=98*u.deg, dec0=-35*u.deg, d0=8*u.kpc, pmra0=10*u.mas/u.yr, pmdec0=-8*u.mas/u.yr, vr0=250*u.km/u.s, tstream=30*u.Myr, fra=True, provenance=[4,2,np.nan])

    props['gjoll'] = dict(label='Gj\\"{o}ll', wangle=360*u.deg, ra0=90*u.deg, dec0=-21*u.deg, d0=3.5*u.kpc, pmra0=24*u.mas/u.yr, pmdec0=-22*u.mas/u.yr, vr0=150*u.km/u.s, tstream=13*u.Myr, fra=True, provenance=[4,2,np.nan])
    
    props['fjorm'] = dict(label='Fj\\"{o}rm', wangle=360*u.deg, ra0=260*u.deg, dec0=70*u.deg, d0=5*u.kpc, pmra0=6*u.mas/u.yr, pmdec0=3*u.mas/u.yr, vr0=-100*u.km/u.s, tstream=30*u.Myr, fra=True, provenance=[4,2,np.nan])
    
    props['fimbulthul'] = dict(label='Fimbulthul', wangle=360*u.deg, ra0=198*u.deg, dec0=-32*u.deg, d0=4*u.kpc, pmra0=-9*u.mas/u.yr, pmdec0=-9*u.mas/u.yr, vr0=250*u.km/u.s, tstream=20*u.Myr, fra=True, provenance=[4,2,np.nan])
    
    props['ylgr'] = dict(label='Ylgr', wangle=360*u.deg, ra0=183*u.deg, dec0=-38*u.deg, d0=9*u.kpc, pmra0=-0.5*u.mas/u.yr, pmdec0=-5*u.mas/u.yr, vr0=320*u.km/u.s, tstream=30*u.Myr, fra=True, provenance=[4,2,np.nan])

    props['sylgr'] = dict(label='Sylgr', wangle=360*u.deg, ra0=164*u.deg, dec0=-13*u.deg, d0=4*u.kpc, pmra0=-25*u.mas/u.yr, pmdec0=-22*u.mas/u.yr, vr0=-200*u.km/u.s, tstream=15*u.Myr, fra=True, provenance=[4,2,np.nan])
    
    props['slidr'] = dict(label='Slidr', wangle=360*u.deg, ra0=148*u.deg, dec0=17*u.deg, d0=3.5*u.kpc, pmra0=-28*u.mas/u.yr, pmdec0=-10*u.mas/u.yr, vr0=-50*u.km/u.s, tstream=20*u.Myr, fra=True, provenance=[4,2,np.nan])

    props['phlegethon'] = dict(label='Phlegethon', wangle=360*u.deg, ra0=299*u.deg, dec0=-61*u.deg, d0=3.5*u.kpc, pmra0=-12*u.mas/u.yr, pmdec0=-25*u.mas/u.yr, vr0=250*u.km/u.s, tstream=60*u.Myr, fra=False, provenance=[8,2,np.nan])
    
    props['phoenix'] = dict(label='Phoenix', wangle=360*u.deg, ra0=27.5*u.deg, dec0=-44*u.deg, d0=16*u.kpc, pmra0=2.8*u.mas/u.yr, pmdec0=-0.2*u.mas/u.yr, vr0=0*u.km/u.s, tstream=30*u.Myr, fra=True, provenance=[1,2,np.nan])
    
    props['turranburra'] = dict(label='Turranburra', wangle=360*u.deg, ra0=59*u.deg, dec0=-18*u.deg, d0=10*u.kpc, pmra0=0.35*u.mas/u.yr, pmdec0=-1.2*u.mas/u.yr, vr0=0*u.km/u.s, tstream=60*u.Myr, fra=True, provenance=[1,2,np.nan])
    
    props['indus'] = dict(label='Indus', wangle=360*u.deg, ra0=352*u.deg, dec0=-65*u.deg, d0=16*u.kpc, pmra0=4.5*u.mas/u.yr, pmdec0=-4.5*u.mas/u.yr, vr0=-10*u.km/u.s, tstream=60*u.Myr, fra=True, provenance=[1,2,np.nan])

    props['elqui'] = dict(label='Elqui', wangle=360*u.deg, ra0=10*u.deg, dec0=-36*u.deg, d0=30*u.kpc, pmra0=0.1*u.mas/u.yr, pmdec0=-0.5*u.mas/u.yr, vr0=-150*u.km/u.s, tstream=100*u.Myr, fra=True, provenance=[1,2,np.nan])
    
    props['jhelum'] = dict(label='Jhelum', wangle=180*u.deg, ra0=4*u.deg, dec0=-52*u.deg, d0=10*u.kpc, pmra0=8*u.mas/u.yr, pmdec0=-3*u.mas/u.yr, vr0=-50*u.km/u.s, tstream=30*u.Myr, fra=True, provenance=[1,2,np.nan])

    props['atlas'] = dict(label='ATLAS', wangle=180*u.deg, ra0=9*u.deg, dec0=-20*u.deg, d0=18*u.kpc, pmra0=-0.5*u.mas/u.yr, pmdec0=-1*u.mas/u.yr, vr0=-150*u.km/u.s, tstream=60*u.Myr, fra=True, provenance=[1,2,3])
    
    props['aliqa_uma'] = dict(label='Aliqa Uma', wangle=180*u.deg, ra0=31*u.deg, dec0=-32*u.deg, d0=26*u.kpc, pmra0=0.25*u.mas/u.yr, pmdec0=-0.7*u.mas/u.yr, vr0=-60*u.km/u.s, tstream=40*u.Myr, fra=True, provenance=[1,2,3])
    
    props['ravi'] = dict(label='Ravi', wangle=360*u.deg, ra0=344.1*u.deg, dec0=-59*u.deg, d0=25*u.kpc, pmra0=0.9*u.mas/u.yr, pmdec0=-2.5*u.mas/u.yr, vr0=100*u.km/u.s, tstream=130*u.Myr, fra=True, provenance=[1,2,np.nan])
    
    props['turbio'] = dict(label='Turbio', wangle=360*u.deg, ra0=27.8*u.deg, dec0=-45*u.deg, d0=16*u.kpc, pmra0=2.*u.mas/u.yr, pmdec0=2*u.mas/u.yr, vr0=100*u.km/u.s, tstream=20*u.Myr, fra=False, provenance=[1,2,np.nan])
    
    props['wambelong'] = dict(label='Wambelong', wangle=360*u.deg, ra0=91*u.deg, dec0=-46*u.deg, d0=16*u.kpc, pmra0=2*u.mas/u.yr, pmdec0=-1*u.mas/u.yr, vr0=150*u.km/u.s, tstream=100*u.Myr, fra=True, provenance=[1,2,np.nan])
    
    props['willka_yaku'] = dict(label='Willka Yaku', wangle=360*u.deg, ra0=38.5*u.deg, dec0=-58*u.deg, d0=41*u.kpc, pmra0=1*u.mas/u.yr, pmdec0=0.5*u.mas/u.yr, vr0=-50*u.km/u.s, tstream=40*u.Myr, fra=True, provenance=[1,2,np.nan])
    
    props['triangulum'] = dict(label='Triangulum', wangle=360*u.deg, ra0=21.2*u.deg, dec0=35*u.deg, d0=28*u.kpc, pmra0=0.8*u.mas/u.yr, pmdec0=0.3*u.mas/u.yr, vr0=-68*u.km/u.s, tstream=70*u.Myr, fra=False)

    return props[name]

def test(name, save_ext='', dra=2, best=True):
    """"""
    stream = Stream(name, save_ext=save_ext)
    
    if best:
        res = pickle.load(open('../data/fits/minimization_{:s}.pkl'.format(stream.savename), 'rb'))
        p0 = [x*y.unit for x, y in zip(res.x, stream.p0)]
        dec, dist, pmra, pmdec, vr = p0
        print(p0)
        fit_label = 'Best-fit'
    else:
        dec, dist, pmra, pmdec, vr = stream.p0
        fit_label = 'Initial'
    
    c = coord.SkyCoord(ra=stream.ra0*u.deg, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vr, frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)

    orbit = stream.ham.integrate_orbit(w0, dt=stream.dt, n_steps=stream.nstep)
    model = orbit.to_coord_frame(coord.ICRS, galactocentric_frame=stream.gc_frame)
    
    # determine orientation
    if stream.fra:
        model_x = model.ra.wrap_at(stream.wangle)
        model_y = model.dec
        ix = 0
        iy = 1
        xlabel = 'R.A. [deg]'
        ylabel = 'Dec [deg]'
    else:
        model_x = model.dec
        model_y = model.ra.wrap_at(stream.wangle)
        ix = -1
        #iy = 0
        tmp = stream.data['dec'][1]
        stream.data['dec'][1] = stream.data['dec'][0]
        stream.data['dec'][0] = tmp
        xlabel = 'Dec [deg]'
        ylabel = 'R.A. [deg]'
    
    # plot data
    plt.close()
    fig, ax = plt.subplots(5, 1, figsize=(7,11), sharex=True)

    fields = ['dec', 'dist', 'pmra', 'pmdec', 'vr']
    labels = [ylabel, 'Distance [kpc]', '$\mu_\\alpha$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]',
            '$V_r$ [km s$^{-1}$]']
    model_fields = [model_y, model.distance, model.pm_ra_cosdec, model.pm_dec, model.radial_velocity]
    istart, iend = 0, -1

    for i in range(5):
        plt.sca(ax[i])
        
        if fields[i] in stream.data.keys():
            plt.plot(stream.data[fields[i]][ix], stream.data[fields[i]][1], 'k.', label='Data')
            plt.errorbar(stream.data[fields[i]][ix].value, stream.data[fields[i]][1].value, yerr=stream.data[fields[i]][2].value, fmt='none', color='k', alpha=0.7, label='')
            
        plt.plot(model_x[istart:iend], model_fields[i][istart:iend], '-', color='tab:blue', label='{:s} orbit'.format(fit_label))
        
        plt.ylabel(labels[i])
        if i==0:
            plt.legend(fontsize='small', handlelength=1)

    plt.minorticks_on()
    plt.xlim(np.min(stream.data['dec'][0].to(u.deg).value)-dra, np.max(stream.data['dec'][0].to(u.deg).value)+dra)
    plt.xlabel(xlabel)

    plt.tight_layout(h_pad=0)
    if best:
        plt.savefig('../plots/diag/best_{:s}.png'.format(stream.savename))
        pickle.dump(c, open('../data/fits/orbit_{:s}.pkl'.format(stream.savename), 'wb'))

def fit_stream(name):
    """"""
    
    stream = Stream(name, ham=ham, save_ext='')
    res = stream.orbit_minimize(save=True)
    stream.orbital_properties(save=True)
    
    t = Table.read('../data/fits/minimization_orbit_{:s}.fits'.format(name))
    t.pprint()
    
    stream = Stream(name, ham=ham_tilt, save_ext='tilt')
    res = stream.orbit_minimize(save=True)
    stream.orbital_properties(save=True)
    
    t = Table.read('../data/fits/minimization_orbit_{:s}_tilt.fits'.format(name))
    t.pprint()


# Globular clusters

def gc_best():
    """Update Vasiliev 2019 GC positions w EDR3 PMs"""
    
    t = Table.read('/home/ana/data/gc_vb2021.fits.gz')
    t.pprint()
    
    to = Table.read('/home/ana/data/gc_vasiliev.fits')
    to = to[to['Dist']>0]
    to.pprint()
    
    to['plx'] = Column(name='plx', dtype=t.dtype['plx'], unit=t['plx'].unit, length=len(to))
    to['e_plx'] = Column(name='e_plx', dtype=t.dtype['e_plx'], unit=t['e_plx'].unit, length=len(to))
    cols = ['pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'plx', 'e_plx']
    
    for e in range(len(to)):
        name = to['Name'][e]
        ind = np.array([True if name in x else False for x in t['Name']])
    
        for c in cols:
            to[c][e] = t[c][ind]
    
    
    
    # adopted parallax
    # Gaia EDR3 parallax if uncertainty below 15%, otherwise Vasiliev 2019 distance, and assuming 15% uncertainty
    f_err = to['e_plx']/np.abs(to['plx'])
    ind_good = f_err<=0.15
    
    to['plx_adpt'] = Column(name='plx_adpt', dtype=t.dtype['plx'], unit=t['plx'].unit, length=len(to))
    to['e_plx_adpt'] = Column(name='e_plx_adpt', dtype=t.dtype['e_plx'], unit=t['e_plx'].unit, length=len(to))
    
    for c in ['plx', 'e_plx']:
        to['{:s}_adpt'.format(c)][ind_good] = to[c][ind_good]
    
    to['plx_adpt'][~ind_good] = to['Dist'][~ind_good].to(u.mas, equivalencies=u.parallax())
    to['e_plx_adpt'][~ind_good] = 0.15 * to['Dist'][~ind_good].to(u.mas, equivalencies=u.parallax())
    
    to.round(4)
    to.write('/home/ana/data/gc_best.fits', overwrite=True)
    to.pprint()
    
    plt.close()
    plt.figure()
    
    plt.scatter(to['Dist'], np.abs(to['plx']**-1), c=f_err, norm=mpl.colors.LogNorm(vmax=0.15))
    plt.plot(to['Dist'], to['Dist'], 'r-')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    #plt.xlim(1,100)
    #plt.ylim(1,100)
    
    plt.tight_layout()

def gap_orbit():
    """Orbital point of the GD-1 gap at phi_1~-40deg"""
    
    stream = Stream('gd1', ham=ham_tilt, save_ext='tilt')
    c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    orbit = stream.ham.integrate_orbit(w0, dt=stream.dt, n_steps=stream.nstep)
    model = orbit.to_coord_frame(gc.GD1, galactocentric_frame=stream.gc_frame)
    
    # gap location (see notebook)
    phi1_gap = -40.75*u.deg
    fwhm = 0.5*5.5*u.deg
    sigma = fwhm / 2.35

    plt.close()
    plt.figure(figsize=(12,6))
    
    plt.plot(model.phi1, model.phi2, 'k-')
    plt.axvline(phi1_gap.value, color='r')
    plt.axvline(-38, color='r')
    plt.axvline(-43.5, color='r')
    
    Nsample = 1000
    np.random.seed(2842)
    pgap = phi1_gap + np.random.randn(Nsample)*sigma
    for i in range(Nsample):
        plt.axvline(pgap[i].value, color='b', lw=1, alpha=0.05, zorder=0)
    
    plt.xlim(-100,20)
    plt.ylim(-10,10)
    plt.gca().set_aspect('equal')
    plt.tight_layout()

def gap_gc_sampling(name='NGC 1851', Nsample=1000, pot='fiducial'):
    """"""
    
    if pot=='tilt':
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    elif pot=='tilt_alt':
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    else:
        ham = ham_fiducial
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
        pot = 'fiducial'
    
    np.random.seed(2925)
    
    ##########
    # gap now
    
    # orbit in observables
    stream = Stream('gd1', ham=ham, save_ext=pot)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=stream.dt, n_steps=stream.nstep)
    model = orbit.to_coord_frame(gc.GD1, galactocentric_frame=stream.gc_frame)
    
    # interpolate
    model_phi1 = model.phi1.wrap_at(180*u.deg).to(u.deg).value
    model_phi2 = model.phi2.to(u.deg).value
    model_dist = model.distance.to(u.kpc).value
    model_pmphi1 = model.pm_phi1_cosphi2.to(u.mas/u.yr).value
    model_pmphi2 = model.pm_phi2.to(u.mas/u.yr).value
    model_vr = model.radial_velocity.to(u.km/u.s).value
    
    order = 3
    bbox = [-180, 180]
    ix = np.argsort(model_phi1)
    model_phi1 = model_phi1[ix]
    
    interp = {}
    interp['phi2'] = InterpolatedUnivariateSpline(model_phi1, model_phi2[ix], k=order, bbox=bbox)
    interp['dist'] = InterpolatedUnivariateSpline(model_phi1, model_dist[ix], k=order, bbox=bbox)
    interp['pmphi1'] = InterpolatedUnivariateSpline(model_phi1, model_pmphi1[ix], k=order, bbox=bbox)
    interp['pmphi2'] = InterpolatedUnivariateSpline(model_phi1, model_pmphi2[ix], k=order, bbox=bbox)
    interp['vr'] = InterpolatedUnivariateSpline(model_phi1, model_vr[ix], k=order, bbox=bbox)
    
    # samples
    phi1_gap = -40.75*u.deg
    fwhm = 0.5*5.5*u.deg
    sigma = fwhm / 2.35
    pgap = phi1_gap + np.random.randn(Nsample)*sigma
    
    gap_coord = {}
    for k in interp.keys():
        gap_coord[k] = interp[k](pgap.value)
    
    c_gap = coord.SkyCoord(phi1=pgap, phi2=gap_coord['phi2']*u.deg, distance=gap_coord['dist']*u.kpc, pm_phi1_cosphi2=gap_coord['pmphi1']*u.mas/u.yr, pm_phi2=gap_coord['pmphi2']*u.mas/u.yr, radial_velocity=gap_coord['vr']*u.km/u.s, frame=gc.GD1)
    w0_gap = gd.PhaseSpacePosition(c_gap.transform_to(gc_frame).cartesian)
    
    
    #########
    # GC now
    
    tgc = Table.read('../data/gc_best.fits')
    ind = [True if name==name_ else False for name_ in tgc['Name']]
    t = tgc[ind]
    #print(t)
    
    gc_coord = {}
    ckeys = ['plx_adpt', 'pmRA', 'pmDE', 'HRV']
    for k in ckeys:
        gc_coord[k] = (t[k] + np.random.randn(Nsample)*t['e_{:s}'.format(k)]) * t[k].unit
    
    for k in ['RAdeg', 'DEdeg']:
        gc_coord[k] = np.repeat(t[k], Nsample)
    
    c_gc = coord.SkyCoord(ra=gc_coord['RAdeg'], dec=gc_coord['DEdeg'], distance=gc_coord['plx_adpt'].to(u.kpc, equivalencies=u.parallax()), pm_ra_cosdec=gc_coord['pmRA'], pm_dec=gc_coord['pmDE'], radial_velocity=gc_coord['HRV'], frame='icrs')
    w0_gc = gd.PhaseSpacePosition(c_gc.transform_to(gc_frame).cartesian)
    
    
    #####################
    # Relative distances
    
    dt = -0.5*u.Myr
    T = 1*u.Gyr
    nstep = np.int(np.abs(T/dt).decompose())
    
    # gap orbits
    orbit_gap = [None] * Nsample
    pos_gap = [None] * Nsample
    
    # gc orbits
    orbit_gc = [None] * Nsample
    pos_gc = [None] * Nsample
    dist = np.empty((Nsample, nstep+1))
    lz = np.empty(Nsample) * u.kpc**2 * u.Myr**-1
    etot = np.empty(Nsample) * u.kpc**2 * u.Myr**-2
    
    for i in range(Nsample):
        orbit_gap[i] = ham.integrate_orbit(w0_gap[i], dt=dt, n_steps=nstep)
        pos_gap[i] = np.array([orbit_gap[i].pos.x, orbit_gap[i].pos.y, orbit_gap[i].pos.z])

        orbit_gc[i] = ham.integrate_orbit(w0_gc[i], dt=dt, n_steps=nstep)
        pos_gc[i] = np.array([orbit_gc[i].pos.x, orbit_gc[i].pos.y, orbit_gc[i].pos.z])
    
        dist[i] = np.linalg.norm(pos_gap[i] - pos_gc[i], axis=0)
        lz[i] = orbit_gc[i].angular_momentum()[2,0]
        etot[i] = orbit_gc[i].energy()[0]
    
    # minimum distance
    ind_min = np.argmin(dist, axis=1)
    min_dist = np.min(dist, axis=1)
    min_t = orbit_gap[0].t[ind_min]
    print(np.min(dist))
    
    ind_close = min_dist < 0.5
    f_close = np.sum(ind_close)/Nsample

    outdict = dict(w0=w0_gc, dmin=min_dist, tmin=min_t, lz=lz, etot=etot)
    pickle.dump(outdict, open('../data/gc_impact/impact_{:s}_{:s}.pkl'.format(pot, name),'wb'))


    # plotting
    plt.close()
    plt.figure(figsize=(8,8))
    
    #plt.scatter(min_t, min_dist, c=pgap.value)
    plt.scatter(min_t, min_dist, c=lz.value, vmin=np.percentile(lz.value, 5), vmax=np.percentile(lz.value, 95))
    #plt.scatter(min_t, min_dist, c=etot.value, vmin=np.percentile(etot.value, 5), vmax=np.percentile(etot.value, 95))
    
    plt.gca().set_yscale('log')
    plt.xlabel('Time [Myr]')
    plt.ylabel('Minimum distance [kpc]')
    plt.title('{:s} f(<500pc) = {:.3}'.format(name, f_close), fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/sampled_gap_{:s}_{:s}.png'.format(pot, name))
    
def sample_gap_gcs(tilt=True):
    """"""
    tgc = Table.read('../data/gc_best.fits')
    
    for e, name in enumerate(tgc['Name'][:]):
        print(e, name)
        gap_gc_sampling(name=name, Nsample=4000, tilt=tilt)
    
def mcmc_gc_present(name='NGC 1851', tilt=False):
    """Sample present-day position of a globular cluster to minimize distance from the GD-1 orbit
    Use measured position as a prior"""
    
    if tilt:
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
        label = 'tilt'
    else:
        ham = ham_fiducial
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
        label = 'fiducial'
    
    dt = -0.5*u.Myr
    T = 1*u.Gyr
    Nstream = 200
    Nstep = np.int(np.abs((T/dt).decompose()))
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=Nstep+Nstream)
    model = orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    orbit_gap = ham.integrate_orbit(w0, dt=dt, n_steps=Nstep+Nstream)
    pos_gap = np.array([orbit_gap.pos.x, orbit_gap.pos.y, orbit_gap.pos.z])
    
    phi1_gap = -40.75*u.deg
    width = 0.07 #kpc
    
    enum = np.arange(np.size(model.phi1), dtype=int)
    ind = np.argmin(np.abs(model.phi1[:Nstream]-phi1_gap))
    i_gap = enum[ind]
    
    
    # globular cluster
    tgc = Table.read('../data/gc_best.fits')
    ind = [True if name==name_ else False for name_ in tgc['Name']]
    t = tgc[ind]
    
    ckeys = ['plx_adpt', 'pmRA', 'pmDE', 'HRV']
    p0s = np.array([t[k][0] for k in ckeys])
    #p0s[0] = 20
    
    nwalkers = 64
    nsteps = 512
    nth = 3
    np.random.seed(4728)
    
    p0 = emcee.utils.sample_ball(p0s, [1e-5, 1e-3, 1e-3, 1e-3], nwalkers)
    p0[:,0] = np.abs(p0[:,0])
    
    x = p0[0]
    
    # prior
    prior = 0
    for e, k in enumerate(ckeys):
        prior += -0.5 * (x[e] - t[k][0])**2 / t['e_{:s}'.format(k)][0]**2 - 0.5*np.log(2*np.pi) - np.log(t['e_{:s}'.format(k)][0])
    print(prior)
    
    #print(x, t['RAdeg'], (x[0]*t['plx_adpt'].unit).to(u.kpc, equivalencies=u.parallax()), x[1]*t['pmRA'].unit, x[2]*t['pmDE'].unit, x[3]*t['HRV'].unit)
    
    t0 = time.time()
    
    c_gc = coord.SkyCoord(ra=t['RAdeg'][0]*t['RAdeg'].unit, dec=t['DEdeg'][0]*t['DEdeg'].unit, distance=(x[0]*t['plx_adpt'].unit).to(u.kpc, equivalencies=u.parallax()), pm_ra_cosdec=x[1]*t['pmRA'].unit, pm_dec=x[2]*t['pmDE'].unit, radial_velocity=x[3]*t['HRV'].unit, frame='icrs')
    w0_gc = gd.PhaseSpacePosition(c_gc.transform_to(gc_frame).cartesian)
    
    orbit_gc = ham.integrate_orbit(w0_gc, dt=dt, n_steps=Nstep)
    pos_gc = np.array([orbit_gc.pos.x, orbit_gc.pos.y, orbit_gc.pos.z])

    dist = np.linalg.norm(pos_gap[:,i_gap:i_gap+Nstep+1] - pos_gc, axis=0)
    chi2 = -0.5 * np.min(dist)**2 / width**2 - 0.5*np.log(2*np.pi) - np.log(width)
    
    t1 = time.time()
    
    print(t1 - t0, phi1_gap, i_gap, np.min(dist), chi2)
    
    #plt.close()
    #plt.figure()
    
    #plt.plot(model.phi1, model.phi2, 'k-')
    
    #plt.tight_layout()


# Impact on the stream

def stream_section(tilt=False):
    """Store positions of stream sections between phi_1 -50 and -30"""
    
    if tilt:
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
        ham = ham_tilt
        label = 'tilt'
    else:
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
        ham = ham_fiducial
        label = 'fiducial'
    
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    dt = -0.01*u.Myr
    tstream = 75*u.Myr
    nstep = np.int(np.abs((tstream/dt).decompose()))
    
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=nstep)
    model = orbit.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    ind = (model.phi1>-50*u.deg) & (model.phi1<-30*u.deg)
    print(np.sum(ind))
    
    wgap = gd.PhaseSpacePosition(pos=orbit[ind].pos, vel=orbit[ind].vel)
    pickle.dump(wgap, open('../data/fits/gap_points_{:s}.pkl'.format(label), 'wb'))
    
    plt.close()
    plt.figure(figsize=(10,5))
    
    plt.plot(model.phi1, model.phi2, 'k-')
    plt.plot(model.phi1[ind], model.phi2[ind], 'ro')
    
    plt.tight_layout()

def compare_gap_sections():
    """Plot orbit sections around the -40 gap in fiducial and tilted potential
    -- within 10arcsec"""
    
    w_fid = pickle.load(open('../data/fits/gap_points_fiducial.pkl', 'rb'))
    w_tilt = pickle.load(open('../data/fits/gap_points_tilt.pkl', 'rb'))
    
    c_fid = w_fid.to_coord_frame(gc.GD1)
    c_tilt = w_tilt.to_coord_frame(gc.GD1)
    
    plt.close()
    plt.figure()
    
    plt.plot(c_fid.phi1, c_fid.phi2, 'o')
    plt.plot(c_tilt.phi1, c_tilt.phi2, 'o')



    plt.tight_layout()

# gc orbits
def gc_orbits():
    """"""
    t = Table.read('../data/gc_best.fits')[:]
    N = len(t)
    
    dt = -1*u.Myr
    T = 5*u.Gyr
    nstep = np.int(np.abs(T/dt).decompose())
    
    c = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['plx_adpt'].to(u.kpc, equivalencies=u.parallax()), pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit = ham_fiducial.integrate_orbit(w0, dt=dt, n_steps=nstep)
    pos = np.array([orbit.pos.x, orbit.pos.y, orbit.pos.z])
    
    #print(orbit.spherical.pos.distance)
    
    rapo = orbit.apocenter()
    rperi = orbit.pericenter()
    print(np.sum((rapo>10*u.kpc)))
    print(np.sum((rapo>10*u.kpc) & (rperi<25*u.kpc)))
    
    t['rapo'] = rapo
    t['rperi'] = rperi
    
    t.write('../data/gc_best_orbits.fits', overwrite=True)
    
    plt.close()
    plt.figure()
    
    plt.hist(rapo.value, bins=np.logspace(0,2.3,50))
    
    plt.gca().set_xscale('log')
    
    plt.tight_layout()

def masses():
    """"""
    t = Table.read('/home/ana/data/baumgardt_gc_combined_table_edited.txt', format='ascii', delimiter=' ')
    t.pprint()
    names = [x.replace('_', ' ') for x in t['Cluster']]
    t['Name'] = names
    t.pprint()
    
    t.write('../data/gc_masses.fits', overwrite=True)

def gc_summary_table():
    """"""
    t = Table.read('../data/gc_best_orbits.fits')
    tm = Table.read('../data/gc_masses.fits')
    
    N = len(t)
    mass = np.zeros(N)
    rm = np.zeros(N)
    
    for i in range(N):
        ind = tm['Name']==t['Name'][i]
        
        m = tm['Mass'][ind]
        if len(m)>0:
            mass[i] = tm['Mass'][ind]
        else:
            mass[i] = np.nan
        
        r = tm['rhm'][ind]
        if len(r)>0:
            rm[i] = tm['rhm'][ind]
        else:
            rm[i] = np.nan
    
    t['mass'] = mass*u.Msun
    t['rhm'] = rm*u.pc
    
    t.write('../data/gc_summary.fits', overwrite=True)

def gc_selection():
    """"""
    t = Table.read('../data/gc_summary.fits')
    ind = (t['rapo']>10) & (t['mass']>1e5*u.Msun)
    print(np.sum(ind))
    
    plt.close()
    plt.figure()
    
    plt.plot(t['rapo'], t['mass'], 'ko')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.xlabel('$r_{apo}$ [kpc]')
    plt.ylabel('M [$M_\odot$]')
    
    plt.tight_layout()


# MCMC
def lnnormal(x, mu, std):
    return -0.5 * (x-mu)**2 / std**2 - 0.5*np.log(2*np.pi) - np.log(std)

def lnprior(p, t):
    """Current observational constraints as Gaussian priors"""
    
    ckeys = ['plx_adpt', 'pmRA', 'pmDE', 'HRV']
    lnp = 0
    for e, k in enumerate(ckeys):
        lnp += lnnormal(p[e], t[k][0], t['e_{:s}'.format(k)][0])
    
    return lnp

def lnlike(p, dt, Nstep, t, pos_gap, i_gap, width):
    """Closest impact"""
    
    c_gc = coord.SkyCoord(ra=t['RAdeg'][0]*t['RAdeg'].unit, dec=t['DEdeg'][0]*t['DEdeg'].unit, 
                          distance=(p[0]*t['plx_adpt'].unit).to(u.kpc, equivalencies=u.parallax()), 
                          pm_ra_cosdec=p[1]*t['pmRA'].unit, pm_dec=p[2]*t['pmDE'].unit, 
                          radial_velocity=p[3]*t['HRV'].unit, frame='icrs')
    w0_gc = gd.PhaseSpacePosition(c_gc.transform_to(gc_frame).cartesian)
    
    orbit_gc = ham_fiducial.integrate_orbit(w0_gc, dt=dt, n_steps=Nstep)
    pos_gc = np.array([orbit_gc.pos.x, orbit_gc.pos.y, orbit_gc.pos.z])

    dist = np.linalg.norm(pos_gap[:,i_gap:i_gap+Nstep+1] - pos_gc, axis=0)
    lnl = lnnormal(np.min(dist), 0, width)
    
    dmin = np.min(dist)
    imin = np.argmin(dist)
    #print(np.size(imin), np.size(dmin), imin, dmin)
    #if np.size(imin)>1:
        ##print(dmin, imin)
        #dmin = dmin[0]
        #imin = imin[0]
    
    return (lnl, dmin, imin)

def lnprob(p, dt, Nstep, t, pos_gap, i_gap, width):
    lnp = lnprior(p, t)
    if not np.all(np.isfinite(lnp)):
        return -np.inf
    
    lnl, dmin, imin = lnlike(p, dt, Nstep, t, pos_gap, i_gap, width)
    if not np.all(np.isfinite(lnl)):
        return -np.inf
    
    #print(lnl + lnp, dmin, imin*dt.value)
    
    return (lnl + lnp, dmin, imin*dt.value)

def run_mcmc(name='NGC 1851', pot='fiducial'):
    """"""
    
    if pot=='tilt':
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    elif pot=='tilt_alt':
        ham = ham_tilt_alt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    else:
        pot = 'fiducial'
        ham = ham_fiducial
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
    
    # setup integration
    dt = -0.05*u.Myr
    T = 2*u.Gyr
    Nstream = 2000
    Nstep = np.int(np.abs((T/dt).decompose()))

    # integrate stream
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit_gap = ham.integrate_orbit(w0, dt=dt, n_steps=Nstep+Nstream)
    pos_gap = np.array([orbit_gap.pos.x, orbit_gap.pos.y, orbit_gap.pos.z])
    model = orbit_gap.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    # find gap
    phi1_gap = -40.75*u.deg
    width = 0.07 #kpc = stream width

    enum = np.arange(np.size(model.phi1), dtype=int)
    ind = np.argmin(np.abs(model.phi1[:Nstream]-phi1_gap))
    i_gap = enum[ind]
    
    # globular cluster
    tgc = Table.read('../data/gc_best.fits')
    ind = [True if name==name_ else False for name_ in tgc['Name']]
    t = tgc[ind]

    ckeys = ['plx_adpt', 'pmRA', 'pmDE', 'HRV']
    p0s = np.array([t[k][0] for k in ckeys])
    
    #print(lnprob(p0s, dt, Nstep, t, pos_gap, i_gap, width))
    
    # setup sampler
    nwalkers = 64
    nsteps = 1024
    nth = 3
    np.random.seed(4728)

    p0 = emcee.utils.sample_ball(p0s, [1e-5, 1e-3, 1e-3, 1e-3], nwalkers)
    p0[:,0] = np.abs(p0[:,0])
    
    dtype = [('dmin', float), ('tmin', float)]
    pool = Pool(nth)
    sampler = emcee.EnsembleSampler(nwalkers, p0.shape[1], log_prob_fn=lnprob, pool=pool, 
                                    args=(dt, Nstep, t, pos_gap, i_gap, width), blobs_dtype=dtype)
    _ = sampler.run_mcmc(p0, nsteps)
    
    pickle.dump(sampler, open('../data/fits/mcmc_dmin_{:s}_{:s}.pkl'.format(pot, name), 'wb'))
    pool.close()

def batch_run(pot='fiducial', verbose=True):
    """"""
    t = Table.read('../data/gc_summary.fits')
    ind = (t['rapo']>10) & (t['mass']>3e5*u.Msun)
    N = np.sum(ind)
    
    for i in range(4,N):
        name_ = t['Name'][ind][i]
        if verbose: print(i, name_, t['mass'][ind][i], t['rhm'][ind][i])
        
        run_mcmc(name=name_, pot=pot)


# Analyze chains

def mcmc_summary(name='NGC 1851', pot='fiducial'):
    """Run nbody simulations of the impact for sampled cluster present-day positions"""
    
    # load sampler
    ndiscard = 128
    sampler = pickle.load(open('../data/fits/mcmc_dmin_{:s}_{:s}.pkl'.format(pot, name), 'rb'))
    lnp = sampler.get_log_prob(flat=True, discard=ndiscard)
    chain = sampler.get_chain(flat=True, discard=ndiscard)
    blobs = sampler.get_blobs(flat=True, discard=ndiscard)
    idx = np.argmax(lnp, axis=0)
    
    print(blobs['dmin'][idx])
    print(blobs['tmin'][idx])
    print(chain[idx])
    
    plt.close()
    plt.figure()
    
    plt.hist(blobs['dmin'], bins=30)
    
    plt.text(0.9, 0.9, name, transform=plt.gca().transAxes, ha='right',va='center', fontsize='medium')
    plt.ylabel('Number')
    plt.xlabel('$d_{min}$ [kpc]')
    
    plt.tight_layout()

def mcmc_chains(name='NGC 1851', pot='fiducial'):
    """Plot chains"""
    
    # load sampler
    sampler = pickle.load(open('../data/fits/mcmc_dmin_{:s}_{:s}.pkl'.format(pot, name), 'rb'))
    
    names = ['$\\varpi$ [mas]', '$\mu_\\alpha$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']

    plt.close()
    fig, axes = plt.subplots(sampler.ndim, figsize=(10,10), sharex=True, squeeze=False)
        
    for k in range(sampler.ndim):
        for walker in sampler.chain[..., k]:
            axes[k][0].plot(walker, marker='', drawstyle='steps-mid', alpha=0.2)

    for i in range(len(names)):
        axes[i][0].set_ylabel(names[i])
                
    fig.tight_layout()

def mcmc_corner(name='NGC 1851', pot='fiducial'):
    """"""
    
    # load sampler
    ndiscard = 512
    sampler = pickle.load(open('../data/fits/mcmc_dmin_{:s}_{:s}.pkl'.format(pot, name), 'rb'))
    chain = sampler.get_chain(flat=True, discard=ndiscard)

    bins = 30
    names = ['$\\varpi$ [mas]', '$\mu_\\alpha$ [mas yr$^{-1}$]', '$\mu_\delta$ [mas yr$^{-1}$]', '$V_r$ [km s$^{-1}$]']
    
    plt.close()
    corner.corner(chain, bins=bins, labels=names, show_titles=True, title_fmt='.3f', title_kwargs=dict(fontsize='small'))

    plt.tight_layout(h_pad=0.01,w_pad=0.01)


def get_gcparams(name):
    """Get mass and half-mass radius, based on Baumgardt compilation
    source data available at https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html"""
    
    t = Table.read('../data/gc_summary.fits')
    ind = t['Name']==name
    
    gc_params = dict(mass=t['mass'][ind][0]*t['mass'].unit, r=t['rhm'][ind][0]*t['rhm'].unit)
    
    return gc_params

def impact_particles(name='NGC 4590', pot='fiducial'):
    """"""
    
    if pot=='tilt':
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    elif pot=='tilt_alt':
        ham = ham_tilt_alt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    else:
        pot = 'fiducial'
        ham = ham_fiducial
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
    
    # read sampler
    ndiscard = 512
    sampler = pickle.load(open('../data/fits/mcmc_dmin_{:s}_{:s}.pkl'.format(pot, name), 'rb'))
    lnp = sampler.get_log_prob(flat=True, discard=ndiscard)
    chain = sampler.get_chain(flat=True, discard=ndiscard)
    blobs = sampler.get_blobs(flat=True, discard=ndiscard)
    idx = np.argmax(lnp, axis=0)
    p = chain[idx]
    
    
    # read in GD-1 today
    w0_stream = pickle.load(open('../data/fits/gap_points_{:s}.pkl'.format(pot), 'rb'))
    
    # globular cluster
    tgc = Table.read('../data/gc_best.fits')
    ind = [True if name==name_ else False for name_ in tgc['Name']]
    t = tgc[ind]
    c_gc = coord.SkyCoord(ra=t['RAdeg'][0]*t['RAdeg'].unit, dec=t['DEdeg'][0]*t['DEdeg'].unit, 
                          distance=(p[0]*t['plx_adpt'].unit).to(u.kpc, equivalencies=u.parallax()), 
                          pm_ra_cosdec=p[1]*t['pmRA'].unit, pm_dec=p[2]*t['pmDE'].unit, 
                          radial_velocity=p[3]*t['HRV'].unit, frame='icrs')
    w0_cluster = gd.PhaseSpacePosition(c_gc.transform_to(gc_frame).cartesian)

    
    # setup integration
    Tback = 1*u.Gyr
    dt = -0.5*u.Myr
    Nback = np.abs(np.int((Tback/dt).decompose()))
    N = np.size(w0_stream.pos)
    
    # Cluster potential
    gc_params = get_gcparams(name)
    gc_pot = gp.PlummerPotential(m=gc_params['mass'], b=gc_params['r'], units=galactic)
    
    # integrate orbits back
    w0 = gd.combine((w0_cluster, w0_stream))
    particle_pot = [None] * (N + 1)
    nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
    orbits_back = nbody.integrate_orbit(dt=dt, n_steps=Nback)
    
    # integrate forward
    w1 = orbits_back[-1,:]
    particle_pot = [None] * (N + 1)
    particle_pot[0] = gc_pot
    nbody = DirectNBody(w1, particle_pot, external_potential=ham.potential)
    orbits_fwd = nbody.integrate_orbit(dt=-dt, n_steps=Nback)
    
    orbit_stream = orbits_fwd[:,1:]
    stream = orbit_stream[-1].to_coord_frame(gc.GD1)
    #stream = orbits_fwd.to_coord_frame(gc.GD1)
    
    pickle.dump(stream, open('../data/nbody_impact_{:s}_{:s}.pkl'.format(pot, name), 'wb'))
    
    color_by = [stream.radial_velocity, stream.pm_phi1_cosphi2, stream.pm_phi2]
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(12,10), sharex=True, sharey=True)
    
    for i in range(3):
        plt.sca(ax[i])
        #plt.plot(stream.phi1, stream.phi2, 'k.')
        plt.scatter(stream.phi1, stream.phi2, c=color_by[i], cmap='magma')
        plt.ylabel('$\phi_2$ [deg]')
    
    plt.ylim(-2,2)
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.tight_layout(h_pad=0)

def plot_impact(name='NGC 1851', pot='fiducial', hspace=0):
    """"""
    
    stream = pickle.load(open('../data/nbody_impact_{:s}_{:s}.pkl'.format(pot, name), 'rb'))
    stream_eq = stream.transform_to(coord.ICRS)
    
    color_by = [stream.radial_velocity, stream.pm_phi1_cosphi2, stream.pm_phi2]
    labels = ['$V_r$ [km s$^{-1}$]', '$\mu_{\phi_1}$ [mas yr$^{-1}$]', '$\mu_{\phi_2}$ [mas yr$^{-1}$]']
    color_by = [stream_eq.radial_velocity, stream_eq.pm_ra_cosdec, stream_eq.pm_dec]
    labels = ['$V_r$ [km s$^{-1}$]', '$\mu_{\\alpha}$ [mas yr$^{-1}$]', '$\mu_{\delta}$ [mas yr$^{-1}$]']

    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(12,10), sharex=True, sharey=True)
    
    for i in range(3):
        plt.sca(ax[i])
        im = plt.scatter(stream.phi1, stream.phi2, c=color_by[i], cmap='magma')
        plt.ylabel('$\phi_2$ [deg]')
        
        pos = plt.gca().get_position()
        cax = plt.axes([pos.x1+0.02, pos.y0, 0.03, pos.y1-pos.y0])
        plt.colorbar(im, cax=cax, label=labels[i])
        
    plt.sca(ax[2])
    plt.ylim(-2,2)
    plt.xlabel('$\phi_1$ [deg]')
    
    plt.sca(ax[0])
    plt.title(pot, fontsize='medium')

    #plt.tight_layout(h_pad=0)
    plt.savefig('../plots/nbody_impact_{:s}_{:s}.png'.format(pot, name))




def mcmc_generate_gallery(name='NGC 1851', pot='fiducial', Nsample=10, dmax=0.02, verbose=True):
    """Create nbody models of the sampled impacts"""
    
    if pot=='tilt':
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    elif pot=='tilt_alt':
        ham = ham_tilt_alt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
    else:
        pot = 'fiducial'
        ham = ham_fiducial
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
    
    # read in GD-1 today
    w0_stream = pickle.load(open('../data/fits/gap_points_{:s}.pkl'.format(pot), 'rb'))
    
    # globular cluster
    tgc = Table.read('../data/gc_best.fits')
    ind = [True if name==name_ else False for name_ in tgc['Name']]
    t = tgc[ind]
    
    # read sampler
    ndiscard = 512
    sampler = pickle.load(open('../data/fits/mcmc_dmin_{:s}_{:s}.pkl'.format(pot, name), 'rb'))
    lnp = sampler.get_log_prob(flat=True, discard=ndiscard)
    chain = sampler.get_chain(flat=True, discard=ndiscard)
    blobs = sampler.get_blobs(flat=True, discard=ndiscard)
    ind = blobs['dmin']<dmax
    chain = chain[ind,:]
    #idx = np.argmax(lnp, axis=0)
    #p = chain[idx]
    
    # setup integration
    Tback = 1*u.Gyr
    dt = -0.5*u.Myr
    Nback = np.abs(np.int((Tback/dt).decompose()))
    N = np.size(w0_stream.pos)
    
    gc_params = get_gcparams(name)
    gc_pot = gp.PlummerPotential(m=gc_params['mass'], b=gc_params['r'], units=galactic)
    
    np.random.seed(5195)
    Nchain = np.shape(chain)[0]
    if verbose: print('within {:f} kpc: {:d}'.format(dmax, Nchain))

    if Nsample>Nchain:
        ind = np.arange(Nchain, dtype='int')
        Nsample = Nchain
    else:
        ind = np.random.random_integers(0, Nchain-1, Nsample)
    
    # generate models
    
    for i in range(Nsample):
        if verbose: print(i)
        # sample cluster present-day position
        p = chain[ind[i]]
        c_gc = coord.SkyCoord(ra=t['RAdeg'][0]*t['RAdeg'].unit, dec=t['DEdeg'][0]*t['DEdeg'].unit, 
                            distance=(p[0]*t['plx_adpt'].unit).to(u.kpc, equivalencies=u.parallax()), 
                            pm_ra_cosdec=p[1]*t['pmRA'].unit, pm_dec=p[2]*t['pmDE'].unit, 
                            radial_velocity=p[3]*t['HRV'].unit, frame='icrs')
        w0_cluster = gd.PhaseSpacePosition(c_gc.transform_to(gc_frame).cartesian)
        
        # integrate orbits back
        w0 = gd.combine((w0_cluster, w0_stream))
        particle_pot = [None] * (N + 1)
        nbody = DirectNBody(w0, particle_pot, external_potential=ham.potential)
        orbits_back = nbody.integrate_orbit(dt=dt, n_steps=Nback)
        
        # integrate forward
        w1 = orbits_back[-1,:]
        particle_pot = [None] * (N + 1)
        particle_pot[0] = gc_pot
        nbody = DirectNBody(w1, particle_pot, external_potential=ham.potential)
        orbits_fwd = nbody.integrate_orbit(dt=-dt, n_steps=Nback)
        
        orbit_stream = orbits_fwd[:,1:]
        stream = orbit_stream[-1].to_coord_frame(gc.GD1)
        pickle.dump(stream, open('../data/models/nbody_impact_{:s}_{:s}_{:03d}.pkl'.format(pot, name, i), 'wb'))

def mcmc_plot_gallery(name='NGC 1851', pot='fiducial', Nsample=10):
    """Generate a multi-page pdf with sampled impacts"""
    
    pp = PdfPages('../data/models/nbody_impact_{:s}_{:s}.pdf'.format(pot, name))
    
    for i in range(Nsample):
        try:
            stream = pickle.load(open('../data/models/nbody_impact_{:s}_{:s}_{:03d}.pkl'.format(pot, name, i), 'rb'))
            
            color_by = [stream.radial_velocity, stream.pm_phi1_cosphi2, stream.pm_phi2]
            labels = ['$V_r$ [km s$^{-1}$]', '$\mu_{\phi_1}$ [mas yr$^{-1}$]', '$\mu_{\phi_2}$ [mas yr$^{-1}$]']

            plt.close()
            fig, ax = plt.subplots(3,1,figsize=(12,10), sharex=True, sharey=True)
            
            for i in range(3):
                plt.sca(ax[i])
                im = plt.scatter(stream.phi1, stream.phi2, c=color_by[i], cmap='magma', rasterized=True)
                plt.ylabel('$\phi_2$ [deg]')
                
                pos = plt.gca().get_position()
                cax = plt.axes([pos.x1+0.02, pos.y0, 0.03, pos.y1-pos.y0])
                plt.colorbar(im, cax=cax, label=labels[i])
                
            plt.sca(ax[2])
            plt.ylim(-2,2)
            plt.xlabel('$\phi_1$ [deg]')
            
            plt.sca(ax[0])
            plt.title(pot, fontsize='medium')
            
            pp.savefig(fig)
            
        except FileNotFoundError:
            pass
    
    pp.close()
        





#############
# Validation
# search for the cluster's stream

def best_obs_sky(name='NGC 1851', pot='fiducial'):
    """Compare orbits on the sky of the observed and best-fit globular cluster orbit"""
    
    if pot=='tilt':
        ham = ham_tilt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
        p = np.array([8.37406633e-02,  2.09812937e+00, -6.07762966e-01,  3.20323350e+02])
        p = np.array([8.37406633e-02,  2.09812937e+00, -6.07762966e-01,  3.20323350e+02])
    elif pot=='tilt_alt':
        ham = ham_tilt_alt
        c = pickle.load(open('../data/fits/orbit_gd1_tilt.pkl', 'rb'))
        p = np.array([8.18953945e-02,  2.09589184e+00, -5.98356626e-01,  3.20311343e+02])
        p = np.array([8.18690957e-02,  2.09512243e+00, -5.98397060e-01,  3.20317319e+02])
    else:
        pot = 'fiducial'
        ham = ham_fiducial
        c = pickle.load(open('../data/fits/orbit_gd1.pkl', 'rb'))
        p = np.array([8.23051518e-02,  2.10982450e+00, -6.05911614e-01,  3.20353350e+02])
        p = np.array([8.20806697e-02,  2.10471852e+00, -6.10228427e-01,  3.20291225e+02])
    
    # cluster positions
    name = 'NGC 1851'
    tgc = Table.read('../data/gc_best.fits')
    ind = [True if name==name_ else False for name_ in tgc['Name']]
    t = tgc[ind]
    
    c_best = coord.SkyCoord(ra=t['RAdeg'][0]*t['RAdeg'].unit, dec=t['DEdeg'][0]*t['DEdeg'].unit, distance=(p[0]*t['plx_adpt'].unit).to(u.kpc, equivalencies=u.parallax()), pm_ra_cosdec=p[1]*t['pmRA'].unit, pm_dec=p[2]*t['pmDE'].unit, radial_velocity=p[3]*t['HRV'].unit, frame='icrs')
    w0_best = gd.PhaseSpacePosition(c_best.transform_to(gc_frame).cartesian)
    
    c_obs = coord.SkyCoord(ra=t['RAdeg'], dec=t['DEdeg'], distance=t['plx_adpt'].to(u.kpc, equivalencies=u.parallax()), pm_ra_cosdec=t['pmRA'], pm_dec=t['pmDE'], radial_velocity=t['HRV'], frame='icrs')
    w0_obs = gd.PhaseSpacePosition(c_obs.transform_to(gc_frame).cartesian)
    
    # integrate orbits
    dt = -0.05*u.Myr
    T = 1*u.Gyr
    nstep = np.int(np.abs(T/dt).decompose())
    
    orbit_best = ham.integrate_orbit(w0_best, dt=dt, n_steps=nstep)
    orbit_obs = ham.integrate_orbit(w0_obs, dt=dt, n_steps=nstep)
    
    eq_best = orbit_best.to_coord_frame(coord.ICRS, galactocentric_frame=gc_frame)
    eq_obs = orbit_obs.to_coord_frame(coord.ICRS, galactocentric_frame=gc_frame)
    
    model_best = orbit_best.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    model_obs = orbit_obs.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    # stream orbit
    nstream = 2000
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    orbit_gap = ham.integrate_orbit(w0, dt=dt, n_steps=nstep+nstream)
    eq_gap = orbit_gap.to_coord_frame(coord.ICRS, galactocentric_frame=gc_frame)
    model_gap = orbit_gap.to_coord_frame(gc.GD1, galactocentric_frame=gc_frame)
    
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    plt.sca(ax[0])
    plt.plot(model_gap.phi1, model_gap.phi2, 'k-')
    plt.plot(model_obs.phi1, model_obs.phi2, 'r:')
    plt.plot(model_best.phi1, model_best.phi2, 'r-')
    
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.xlabel('$\phi_1$ [deg]')
    plt.ylabel('$\phi_2$ [deg]')
    
    plt.sca(ax[1])
    plt.plot(eq_gap.ra, eq_gap.dec, 'k-')
    plt.plot(eq_obs.ra, eq_obs.dec, 'r:')
    plt.plot(eq_best.ra, eq_best.dec, 'r-')
    
    plt.xlim(360,0)
    plt.ylim(-90,90)
    plt.xlabel('R.A. [deg]')
    plt.ylabel('Dec [deg]')
    
    
    plt.tight_layout()
    plt.savefig('../plots/orbits_sky_{:s}_{:s}.png'.format(pot, name))
