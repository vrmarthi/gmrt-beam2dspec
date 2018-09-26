import numpy as np
import astropy.io.fits as f
import matplotlib.pylab as pl
import glob as glob
import numpy.fft as ft

def waterfall(dds_t, P0, tsamp, sweep): 
    truebins = P0/tsamp
    nsamples = dds_t.size
    nbins = np.int(np.rint(truebins))
    nperiod = np.int(np.rint(nsamples/nbins))
    shape = [nperiod, nbins]
    wfall = np.zeros(shape)
    tstart = 0
    period=0
    for i in np.arange(nsamples):
        nxtbin = np.int(np.rint(i-period*truebins + period*sweep/nperiod))
        if nxtbin >= nbins:
           print('Pulses corrected : ', period+1, end='\r'),
           nxtbin -= nbins
           period += 1
        wfall[period, nxtbin] = dds_t[i]
    return wfall


def show_waterfall(wfall, P0):
    npulses = wfall.shape[0]
    nbins = wfall.shape[1]
    T = npulses * P0 / 60.0
    extt = (0.0, 1.0, T, 0.0)
    pl.imshow(wfall, extent=extt, cmap='Greys', aspect = 2.0/T)
    pl.colorbar()
    pl.xlabel('Pulse phase')
    pl.ylabel('Time [min]')
    sv=input('Save figure ?')
    if sv == 'y':
        pl.savefig('waterfall.png')
        pl.savefig('waterfall.eps')
#    pl.close()

def get_sweep(wfall):
    zoom = 1
    while zoom == 1:
        pl.imshow(wfall, aspect=0.1, cmap='Greys')
        pl.colorbar()
        pl.xlabel('Bins')
        pl.ylabel('Time')
        click = input('Ready to click ? :')
        if click == 'y':
            pl.title('Click on peak for t=0')
            t0 = pl.ginput(1)
            pl.close()
            break
        else:
            continue
    while zoom == 1:
        pl.imshow(wfall, aspect=0.1, cmap='Greys')
        pl.xlabel('Bins')
        pl.ylabel('Time')
        click = input('Ready to click ? :')
        if click == 'y':
            pl.title('Click on peak for t=1')
            t1 = pl.ginput(1)
            pl.close()
            break
        else:
            continue
    sweep = np.int(np.rint(t0[0][0]) - np.rint(t1[0][0]))
    return sweep


def find_sweep(wfall):
    nperiod=wfall.shape[0]
    nbin = wfall.shape[1]
    firstpeak = np.argmax(wfall[0,:])
    lastpeak = np.argmax(wfall[-2,:])
    swp = 1.0*(lastpeak-firstpeak)/(nperiod-1)
    return sweep

def phase_align(dds, P0, tsamp, sweep):
    truebins = P0/tsamp
    nsamples = dds.shape[0]
    nbins = np.int(np.rint(truebins))
    nperiod = np.int(np.rint(nsamples/nbins))
    newdat = np.zeros(dds.shape)
    period=0
    for i in np.arange(nsamples):
        offset = np.int(np.rint(i-period*truebins + period*sweep/nperiod))
        ibin = period*nbins + offset
        if ibin-period*nbins >= nbins:
           print('Pulses corrected : ', period, end='\r'),
           period += 1
        newdat[ibin,:] = dds[i, :]
    return newdat

def dedisperse(file, DM, f_start, f_stop, nchan, tsamp):
    bytes_per_sample = 2
    fh = open(file)
    fh.seek(0,2)
    nsamples = np.int(fh.tell() / (bytes_per_sample * nchan))
    fh.seek(0,0)
    buffershape = [nsamples, nchan]
    print('Raw data file           : ', file)
    print('Samples in file         : ', nsamples)
    print('Sample interval         : ', tsamp*1e3, ' ms')
    print('Channels                : ', nchan)

    bw = f_stop - f_start
    f_ref = (f_start + 0.5 * (bw/nchan))
    fch = f_ref + np.arange(nchan)*(bw/nchan)
    t_delay = 4.149e3 * DM * (1./f_ref**2 - 1./fch**2)

    print('Centre of first channel : ',fch[0], ' MHz')
    print('Centre of last channel  : ',fch[nchan-1], ' MHz')

    buffer = (np.fromfile(fh, dtype='uint16', count=-1)<<3).reshape(buffershape)
    dds = np.ndarray(buffershape, dtype=float)
    for ch in np.arange(nchan):
        print('Channels dedispersed    : ', ch+1, end='\r')
        dds[:,ch] = np.roll(buffer[:,ch], np.int(np.rint(t_delay[ch]/tsamp))).astype(float)
    
    #outfile = file + 'dds.npy'
    #np.save(outfile, dds)
    fh.close()
    return dds

#def collapse_channels(dds, file_to_save):
def collapse_channels(dds):
    dds_t = dds.mean(axis=1)
    #file_to_save = file_to_save + 'dds.single-pulse.npy'
    #np.save(file_to_save, dds_t)
    return dds_t

def nearest_power_of_2(number):
    x = np.log2(number)
    y = np.ceil(x)
    return np.int(2**y)

def get_Period(dds_t, tsamp):
    N = np.int(nearest_power_of_2(dds_t.size))
    print('Padding ', dds_t.size, 'length array with ', N-dds_t.size, 'zeros.')
    DDS_T = np.abs(ft.rfft(dds_t-dds_t.mean(), N))
    nu = ft.rfftfreq(N, tsamp)
    zoom = 1
    while zoom == 1:
        pl.plot(nu, DDS_T)
        pl.xlabel('Frequency [Hz]')
        pl.ylabel('Flux')
        click = input('Ready to click ? :')
        if click == 'y':
            pl.title('Click on peak for F0')
            F0 = pl.ginput(1)
            zoom = 0
            break
        else:
            continue
            
    F0 = F0[0][0]
    P0 = 1.0 / F0
    pl.close()
    
    pl.plot(nu, DDS_T)
    pl.xlim(0.0, 12.5*F0)
    pl.xlabel('Frequency [Hz]')
    pl.ylabel('Flux')
    pl.title('First 12 harmonics')
    pl.savefig('First-12-harmonics.eps')
    pl.savefig('First-12-harmonics.png')
    return P0, F0

def find_gate(wfall)
# works only for B1133+16 with tsamp=2.6ms
    profile=wfall.mean(0)
    peak_bin=np.where(np.max(profile))
    g_on = [np.int(peak-6), np.int(peak+16)]
    return g_on

def select_gate(wfall):
    g_on = [0, 0]
    profile = wfall.mean(axis=0)
    pl.plot(profile)
    zoom = 1
    while zoom == 1:
        pl.plot(profile,'kx', lw=1.0)
        pl.xlabel('Bin #')
        pl.ylabel('Flux')
        click = input('Ready to click ? :')
        if click == 'y':
            pl.title('Select gate for dynamic spectrum')
            gate = pl.ginput(2)
            zoom = 0
            break
        else:
            continue
            
    g_on[0] = np.int(np.floor(gate[0][0]))
    g_on[1] = np.int(np.ceil(gate[1][0]))

    return g_on

def fold(wfall, P0, tsamp, g_on):
    nbins = np.int(np.rint(P0/tsamp))
    off_pulse = (wfall[:,0:g_on[0]].mean() + wfall[:,g_on[1]+1:nbins].mean()) * 0.5
    profile = wfall.mean(axis=0)/off_pulse - 1.0
    profile /= np.max(profile)
    t = np.linspace(0,P0, nbins)
    pl.plot(t,profile, 'k', lw=1.0)
    pl.xlabel('Time [s]')
    pl.ylabel('Amplitude')
    pl.xlim(0,P0)
    pl.ylim(-0.1,1.1)
    pl.savefig('folded_profile.png')
    pl.savefig('folded_profile.eps')
    

def make_dynamic_spectrum(dds, P0, tsamp, g_on):
  stride = np.int(np.rint(P0/tsamp))
  nsamples, nchan = dds.shape
  nbins = np.int(np.rint(P0/tsamp))
  npulses = np.int(nsamples/nbins)
  dynspec = np.zeros([npulses+1, nchan])

  for i in np.arange(npulses):
     print('Processed ', i, ' pulses', end='\r')
     on_start = g_on[0] + stride*i
     on_stop  = g_on[1] + stride*i

     off_start_1 = stride*i
     off_stop_1  = on_start-1

     off_start_2 = on_stop+1
     off_stop_2 = stride*(i+1)
     
     off_pulse = (dds[off_start_1:off_stop_1,:].mean(axis=0) + dds[off_start_2:off_stop_2,:].mean(axis=0)) * 0.5

     dynspec[i,:] = np.mean(dds[on_start:on_stop,:], axis=0)/off_pulse - 1.0

  return dynspec

def parse_das_log(logfile):
    Times=[]
    Timesfile = open('Rec_Blk_times.txt', "w")
    log = open(logfile, 'r')
    for line in log:
        if('#' in line):
            t=(line.split('# ')[1]).split(' ')[0]
            if (t == 'Observation'):
                break
            Times.append(t)
            Timesfile.write(t+'\n')
    Timesfile.close()
    log.close()

    return Times, Timesfile

def update_fits_header(hdu):
 hdu.header['DATE-OBS'] = input('Observation date : \'DD-MMM-YYYY\' : ')
 hdu.header['TELESCOP'] = input('Telescope: ')
 hdu.header['INSTRUME'] = input('Backend used: ')
 hdu.header['DASMODE'] = input('Backend mode: ') 
 hdu.header['PROJECT'] = input('Project code: ')
 hdu.header['STRT-UTC'] = input('Start time UTC:')
 return hdu

def ds_to_fits(data, P, f_start, BW, filename):
 ntime,nchan = data.shape
 hdu = f.PrimaryHDU(data)
 hdu.header['CTYPE1'] ='FREQ - MHZ'
 hdu.header['CTYPE2'] ='TIME - SEC'
 hdu.header['CRVAL1'] = f_start 
 hdu.header['CRVAL2'] = 0.0
 hdu.header['CDELT1'] = 1.0*BW/nchan
 hdu.header['CDELT2'] = P
 hdu.header['CRPIX1'] = 0
 hdu.header['CRPIX2'] = 0 
 hdu.header['CROTA1'] = 0.0
 hdu.header['CROTA2'] = 0.0
 hdu.header['BSCALE'] = 1.0
 hdu.header['DATAMIN'] = np.min(data)
 hdu.header['DATAMAX'] = np.max(data)
 update_fits_header(hdu)
 hdu.writeto(filename)


def ds_to_cs_to_fits(data, P, BW, filename):
 ntime,nchan=data.shape
 new_ntime=nearest_power_of_2(data.shape[0])
 new_nchan = nearest_power_of_2(data.shape[1])
 print(new_ntime, 'X', new_nchan, ' 2D FFT in progress...')
 cs=ft.fftshift(ft.rfft2(data, [new_ntime, new_nchan]),axes=0)
 print('FFT shift...')
 realCS=np.real(cs)
 imagCS=np.imag(cs)
 CS=np.zeros([2,cs.shape[0], cs.shape[1]])
 CS[0,:]=realCS
 CS[1,:]=imagCS
 tau = ft.rfftfreq(new_nchan,np.abs(BW)/nchan)
 fD=ft.fftshift(ft.fftfreq(new_ntime,P))*1e3
 hdu = f.PrimaryHDU(CS)
 hdu.header['CTYPE1'] ='DELAY - MICROSEC'
 hdu.header['CTYPE2'] ='DOPPLER FREQUENCY - MILLIHERTZ'
 hdu.header['CTYPE3'] ='REAL-IMAG'
 hdu.header['CRVAL1'] = tau[0]
 hdu.header['CRVAL2'] = fD[0]
 hdu.header['CRVAL3'] = 0
 hdu.header['CDELT1'] = tau[1]-tau[0]
 hdu.header['CDELT2'] = fD[1]-fD[0]
 hdu.header['CDELT3'] = 0
 hdu.header['CRPIX1'] = 0
 hdu.header['CRPIX2'] = 0
 hdu.header['CRPIX3'] = 0
 hdu.header['CROTA1'] = 0.0
 hdu.header['CROTA2'] = 0.0
 hdu.header['CROTA3'] = 0.0
 hdu.header['BSCALE'] = 1.0
 update_fits_header(hdu)
 hdu.header['INT_TIME']=P*ntime
 hdu.writeto(filename)
 return fD, tau, cs

def make_dynamic_spectrum_cross_beam(dds, P0, tsamp, g_on):
  stride = np.int(np.rint(P0/tsamp))
  nsamples, nchan = dds.shape
  nbins = np.int(np.rint(P0/tsamp))
  npulses = np.int(nsamples/nbins)
  dynspec = np.zeros([npulses+1, nchan])

  for i in np.arange(npulses):
     print('Processed ', i, ' pulses', end='\r')
     on_start = g_on[0] + stride*i
     on_stop  = g_on[1] + stride*i
     dynspec[i,:] = np.mean(dds[on_start:on_stop,:], axis=0)

  return dynspec

def stitch_dspec():                                                           
 b3=sorted(glob.glob('B3*fits'))                                             
 b4=sorted(glob.glob('B4*fits'))                                             
 F0=f.open(b3[0])                                                            
 F1=f.open(b3[-1])                                                           
 G0=f.open(b4[0])
 G1=f.open(b4[-1])
 t0=F0[0].header['STRT-UTC']
 print('First scan starts at : ',t0)
 t1=F1[0].header['STRT-UTC']
 print('Last scan starts at : ', t1)
 t=np.int( ((int(t1.split(':')[0])-int(t0.split(':')[0]))*3600.0 + (int(t1.split(':')[1])-int(t0.split(':')[1]))*60.0 + (float(t1.split(':')[2])-float(t0.split(':')[2])) / 60 ) + F1[0].header['NAXIS2'] )
 print('No. of pulse periods :',  t)
 ds=np.zeros([t+10,16384*2+4096])
 ds[0:G0[0].header['NAXIS2'],0:16384]=G0[0].data
 ds[0:F0[0].header['NAXIS2'],16383+4096:-1]=F0[0].data
 for i in range(1,len(b3)):
   F=f.open(b3[i])
   G=f.open(b4[i])
   t=F[0].header['STRT-UTC']
   offset=np.int( (int(t.split(':')[0])-int(t0.split(':')[0]))*3600.0 + (int(t.split(':')[1])-int(t0.split(':')[1]))*60.0 + (float(t.split(':')[2])-float(t0.split(':')[2])) / 60)
   print('Scan : ',i,'starting at :', t,'with ',offset, 'pulses')
   ds[offset:offset+G[0].data.shape[0],0:16384]=G[0].data
   ds[offset:offset+F[0].data.shape[0],16383+4096:-1]=F[0].data
 return ds
  

# def fit_parabola(CS, outer_a, inner_a, N, f_D, tau):
#   a = np.linspace(outer_a, inner_a, N)
#   XX, NX, YY, NY = 4096, 8, 256, 128
#   cs = (np.abs(CS[:,1:])).reshape([XX, NX, YY, NY]).mean(axis=3).mean(axis=1)
#   flux=np.zeros(N)
#   for i in np.arange(N):
#     fDmax = np.sqrt(tau[-1]/a[i])
#     fdsize = np.int(1.0 * fDmax/f_D[-1] * XX)
#     fD = np.linspace(-fDmax,fDmax, fdsize)
#     t = a[i] * fD**2
#     y = np.zeros(fdsize)
#     mask = np.zeros(cs.shape)
#     for x in np.arange(fdsize):
# 	y = np.int(np.rint(t[x]*(YY-1))/tau[-1])
#         mask[np.int(XX/2 - (fdsize+1)/2 + x), y ] = 1.0
#         mask[np.int(XX/2 - (fdsize+1)/2 + x) - 1, y ] = 1.0
#         mask[np.int(XX/2 - (fdsize+1)/2 + x) + 1, y ] = 1.0
#         mask[np.int(XX/2 - (fdsize+1)/2 + x) - 2, y ] = 1.0
#         mask[np.int(XX/2 - (fdsize+1)/2 + x) + 2, y ] = 1.0
#     flux[i] = (cs*mask).sum()
#     flux[i] = (cs*mask).sum()
#     print(i, a[i], flux[i])
#   return a,flux


def make_gated_dynamic_spectrum(dds, P0, tsamp, g_on, w_cor):
  ngate = g_on[1] - g_on[0] 
  stride = np.int(np.rint(P0/tsamp))
  nsamples, nchan = dds.shape
  nbins = np.int(np.rint(P0/tsamp))
  npulses = np.min(np.int(nsamples/nbins),w_cor.shape[0]) - 2
  dynspec = np.zeros([ngate, npulses+1, nchan])

  for i in np.arange(npulses):
     print('Processed ', i, ' pulses', end='\r')
     on_start = g_on[0] + stride*i
     on_stop  = g_on[1] + stride*i

     off_start_1 = stride*i
     off_stop_1  = on_start-1

     off_start_2 = on_stop+1
     off_stop_2 = stride*(i+1)
     
     off_pulse = (dds[off_start_1:off_stop_1,:].mean(axis=0) + dds[off_start_2:off_stop_2,:].mean(axis=0)) * 0.5

     dynspec[:,i,:] = ((dds[on_start:on_stop,:])/off_pulse - 1.0) / w_cor[i,g_on[0]:g_on[1], None]

  return dynspec
