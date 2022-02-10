"""
USER QuakeCoRE 2021/22
Earthquakes to Audio
Author: Cameron Davis
Version: 1.0.1
Funky: 1.0.0

If any updates are made they will be posted on GitHub at the link below
Updates: https://github.com/Cameron-cpd/eq2audio/releases
"""
"""
# Specify the recording names and folder/event names, for each event list all .000 files then all .090 files
events = ['AF8', 'Kaikoura'] # The earthquake names
recordings = [['Franz Josef.000','Greymouth.000', 'Franz Josef.090','Greymouth.090'], # Recorded ground accelerations for the first earthquake
                ['Kaikoura.000','Wellington.000', 'Kaikoura.090','Wellington.090']] # Recorded ground accelerations for the second earthquake
in_g = False # If the recordings are in units of g put True, if they are in units of cm/s2 put False
"""
make_ani = False # If you want to create an animation for each recording put True
# Creating the animation takes a while so it is best on one recording at a time and only in the area of interest
# Set the variables for animating a time cursor moving across plots of the clipped ground acceleration and averaged accelerations
ani_start = 0 # [s] Set the time to start the animation
ani_dur = 181 # [s] Set the duration of the animation (to animate the entire recording set ani_dur > 180)
ani_step = 0.2 # [s] Set the amount of time to spend on each animation frame


# Single recording
events = ['Random']
recordings = [['Funky (af8 chch.000)']]
in_g = False

"""
# Simulation recordings
events = ['alpinefault', 'christchurch', 'darfield', 'kaikoura']
recordings = [['CastleHill.000','Christchurch.000','FranzJosef.000','Greymouth.000', 'CastleHill.090','Christchurch.090','FranzJosef.090','Greymouth.090'],
                ['CastleHill.000','Christchurch.000','RockDyers.000','SoilCity.000','SoilLyttelton.000', 'CastleHill.090','Christchurch.090','RockDyers.090','SoilCity.090','SoilLyttelton.090'],
                ['CastleHill.000','Christchurch.000', 'CastleHill.090','Christchurch.090'],
                ['CastleHill.000','Christchurch.000','Kaikoura.000','KIKS.000','POTS.000','TEPS.000','Wellington.000', 'CastleHill.090','Christchurch.090','Kaikoura.090','KIKS.090','POTS.090','TEPS.090','Wellington.090']]
in_g = False # The above recordings are in units of cm/s2 so do not need to be converted
"""
"""
# Historic recordings
events = ['kaikoura (historic)', '4.6 (historic)','CHCH (historic)']
recordings = [['KIKS.000','KEKS.000', 'KIKS.090','KEKS.090'],
                ['D14C.000','D14C.090'],
                ['HVSC.000','LPOC.000','REHS.000', 'HVSC.090','LPOC.090','REHS.090']]
in_g = True # The above recordings are in units of g so need to be converted to cm/s2
"""

# Import required modules
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.signal import butter
from scipy.signal import sosfiltfilt
from scipy import interpolate
import mido
import subprocess
import random
import platform

def create_bands():
    """ Generate the centre values for bandpass filtering so that all frequencies between 0.1 and 99.999 are included """
    filtbands = []
    high = 0.8956803352330285 # High pass from the bwfilter function
    low = 1.1164697500474103 # Low pass from the bwfilter function
    filtbands.append(99.999/low)
    while filtbands[-1] > 0.1:
        filtbands.append(filtbands[-1]*high/low)
    filtbands.reverse() # Reverse the order so that the frequencies are in ascending order
    return filtbands

def create_pent(root):
    """ Create a 6 octave (and 1 note) pentatonic scale starting on a given note in MIDI pitch notation """
    pentatonic = []
    for i in np.arange(0,6):
        pentatonic += [root, root+2, root+4, root+7, root+9]
        root += 12
    pentatonic += [root, root+2]
    return pentatonic

def sample_extraction(files):
    """ The sample code from the Seisfinder website to extract the information from the given files. 

    dt: The time step between recorded data points
    vals: The acceleration of the motion at each time step
    """
    dt = []
    vals = []
    for comp_file in files:
        with open(comp_file, 'r') as tseries:
            info1 = tseries.readline().split()
            info2 = tseries.readline().split()

            # sample metadata extraction
            nt = int(info2[0])
            dt.append(float(info2[1]))

            # list() only required in python3
            vals.append(np.array(list(map(float, \
                    ' '.join(map(str.rstrip, tseries.readlines())).split()))))

            # make timeseries start from time = 0
            # t start / dt = number of missing points
            diff = int(round(float(info2[4]) / dt[-1]))
            if diff < 0:
                # remove points before t = 0
                vals[-1] = vals[-1][abs(diff):]
            elif diff > 0:
                # insert zeros between t = 0 and start
                vals[-1] = np.append(np.zeros(diff), vals[-1])
    print('\nData extracted for the {0} earthquake'.format(files[0][:files[0].index('/')]))
    return(dt, vals)

def plot_recordings(eq):
    """ Plot the ground motions with 000 readings in the top row and 090 readings in the bottom row, save the plot as a .png """
    rows = 2 # 1 row per direction
    cols = int(len(eq)/2) # 1 column per site
    if len(eq)==1: # If there is only one recording there is only one row and column
        rows = 1
        cols = 1
    fig = plt.figure(num=1, figsize=(12,8))
    for n in np.arange(len(eq)):
        plot_motion(dts[n], values[n], rows, cols, n+1, eq[n], fig)
    plt.savefig(eq[0][:eq[0].index('/')] + '.png', dpi=300) # Save the figure as a .png
    plt.close()

def plot_motion(dt, vals, row, col, n, title, fig):
    """ Plot the ground motions with 000 readings in the top row and 090 readings in the bottom row """
    fig.suptitle('Recorded Groundmotions for the {0} Earthquake'.format(title[:title.index('/')]))
    ax = fig.add_subplot(row,col,n)
    ax.plot(np.linspace(0, dt*len(vals), len(vals)), vals, color='tab:blue')
    ax.set_title(title[title.index('/')+1:])
    ax.set_xlim(0, dt*len(vals))
    ax.set_xlabel('Time [s]')
    ax.set_ylim(-750, 750)
    ax.set_ylabel(r'Acceleration [cm/s$^2$]')
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False) # Minor ticks on the y-axis
    ax.label_outer()
    fig.tight_layout()

def filter(dt, vals, file):
    """ Use the butterworth filter with bandpass filtering, then plot the resulting waves.
    
    filt_vals: each array is the filtered ground motions for a different bandpass
    filt_bands: each array is the lower and upper edges of the bandpass
    filt_sum: the sum of all the filtered frequencies' accelerations
    vals: the raw ground acceleration clipped to the earthquake event
    """
    # Clip the raw data to the earthquake event as decided by the base value
    max_points = int(180/dt) # Set the maximum number of datapoints as the maximum time in seconds divided by dt
    vals = vals[np.where(vals>base)[0][0]:np.where(vals>base)[0][-1]] # Cut the recorded ground acceleration to the earthquake event
    if len(vals) > max_points:
        vals = vals[0:max_points] # Cut the recorded ground acceleration to a maximum number of datapoints
        vals = vals[np.where(vals>base)[0][0]:np.where(vals>base)[0][-1]] # Cut the recorded ground acceleration to the earthquake event
    
    # Set up the output arrays
    filt_vals = []
    filt_bands = []
    filt_sum  = np.zeros(len(vals))

    # Use the butterworth filter function from QuakeCoRE with each pair of band cutoff values
    for j in bands:
        filt_vals.append(bwfilter(vals, dt, j, 'bandpass', match_powersb=True)[0])
        filt_bands.append(bwfilter(vals, dt, j, 'bandpass', match_powersb=True)[1])
        filt_sum += filt_vals[-1]
    waves[file] = filt_vals # Enter the filtered waves in a dictionary with the filename as the key
    print('\nData filtered for {0}'.format(file))
    return (filt_vals, filt_bands, filt_sum, vals)

def bwfilter(data, dt, freq, band, match_powersb=True):
    """Butterworth filter function from the quakecore github timeseries.py

    data: np.array to filter
    dt: timestep of data
    freq: cutoff frequency
    band: One of {'highpass', 'lowpass', 'bandpass', 'bandstop'}
    match_powersb: shift the target frequency so that the given frequency has no attenuation
    """
    # power spectrum based LF/HF filter (shift cutoff)
    # readable code commented, fast code uncommented
    # order = 4
    # x = 1.0 / (2.0 * order)
    # if band == 'lowpass':
    #    x *= -1
    # freq *= exp(x * math.log(sqrt(2.0) - 1.0))
    nyq = 1.0 / (2.0 * dt)
    highpass_shift = 0.8956803352330285
    lowpass_shift = 1.1164697500474103
    if match_powersb:
        if band == "highpass":
            freq *= highpass_shift
        elif band == "bandpass" or band == "bandstop":
            freq = np.asarray((freq*highpass_shift, freq*lowpass_shift))
        elif band == "lowpass":
            freq *= lowpass_shift
    return [sosfiltfilt(butter(4, freq / nyq, btype=band, output="sos"), data, padtype=None), [round(freq[0], 3), round(freq[1], 3)]]

def plot_filtered(dt, vals, filt_vals, filt_sum, cutoffs, file, title):
    """ Plot the ground acceleration with the approximation and component frequencies. Save the resulting plot as a .png """
    x = np.linspace(0, dt*len(vals), len(vals))
    fig = plt.figure(num=2, figsize=(12,8))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0,1,len(filt_vals))))
    ax = plt.axes()
    ax.plot(x, vals, label='Unfiltered Recording', color='tab:blue') # Plot the unfiltered acceleration
    ax.plot(x, filt_sum, label='Filtered Approximation', dashes=[2,2], linewidth=1, color='darkorange') # Plot the filtered approximaton
    for i in np.arange(len(cutoffs)):
        ax.plot(x, filt_vals[i], label='{0:.3g} to {1:.3g} [Hz]'.format(cutoffs[i][0], cutoffs[i][1]), alpha=0.4) # Plot each component frequency
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(ncol=1, framealpha=1, loc='center left', bbox_to_anchor=(1,0.5))
    ax.set_title(file+'\n'+title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Acceleration [cm/s$^2$]')
    ax.set_xlim(0, dt*len(vals))
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False) # Minor ticks on the y-axis
    plt.savefig(file+' '+title+'.png', dpi=300) # Save the plot as a .png
    plt.close()

def average_amp(record, file, span):
    """ Average the absolute acceleration across the time block. """
    avg = []
    for j in np.arange(len(record)):
        mean = []
        for k in np.arange(span, len(record[j]), span):
            mean.append(np.mean(np.abs(record[j][k-span : k]))) # Find the average acceleration over the block of time
        avg.append(mean)
    averages[file] = avg # Enter the averaged accelerations in a dictionary with the filename as the key

def plot_avg(avg, cutoffs, file, title):
    """ Plot the averaged ground acceleration values that are used to set the audio variables. Save the resulting plot as a .png """
    x = np.arange(block/2, block*len(avg[0]), block)
    fig = plt.figure(num=3, figsize=(12,8))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0,1,len(avg))))
    ax = plt.axes()
    for i in np.arange(len(avg)):
        ax.plot(x, avg[i], label='{0:.3g} to {1:.3g}'.format(cutoffs[i][0], cutoffs[i][1]), alpha=0.4)
    ax.legend(title='Frequency Bandwidths [Hz]', loc=1, framealpha=1, ncol=2)
    ax.set_title(file+'\n'+title)
    ax.set_xlabel('Time [s]')
    ax.set_xlim(0, block*len(avg[0]))
    ax.set_ylabel(r'Mean Acceleration [cm/s$^2$]')
    ax.set_ylim(base,top)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False) # Minor ticks only on the y-axis
    plt.savefig(file+' '+title+'.png', dpi=300) # Save the plot as a .png
    plt.close()

def audio(avg, file):
    """ Create and save a MIDI (.mid) file of the filtered earthquake recording, then convert that to an audio (.flac) file """
    big = False # big will be True if any averaged acceleration in the recording is larger than the maximum
    x = np.linspace(1,1000,10000)
    # Set up note velocity interpolation
    min_vol = 20 # Minimum allowed by MIDI is 0 (0 velocity means the note is silent/released)
    max_vol = 120 # Maximum allowed by MIDI is 127
    vol = 40*np.log(x) + min_vol # Create a logarithmic velocity array
    vol = vol[np.all([min_vol<vol, vol<max_vol], axis=0)] # Clip the velocity array to the max and min volumes
    acc_v = np.linspace(base, top, len(vol)) # Set up a matching array from min to max accelerations
    inter_vol = interpolate.interp1d(acc_v, vol) # Create a linear interpolation function for acceleration to volume
    # Set up note length interpolation
    min_len = 100 # Minimum note length in ticks
    max_len = int(block*960) # Set maximum ticks to the length of the block (default bpm = 120 so 960 ticks per second)
    length = 0.2*x + min_len # Create a linear note length array
    length = length[np.all([min_len<length, length<max_len], axis=0)] # Clip the length array to the max and min lengths
    acc_l = np.linspace(base, top, len(length)) # Set up a matching array from min to max accelerations
    inter_len = interpolate.interp1d(acc_l, length) # Create a linear interpolation function for acceleration to duration

    # Write the midi file using mido
    midi_file = mido.MidiFile() # Create the midi file for the recording
    rec = file+'' # Set the track and file names to the recording name + a string with the identifying features
    for j in np.arange(len(pent)):
        track = mido.MidiTrack() # Create a midi track on the file for each filtered frequency
        midi_file.tracks.append(track) # Attach the midi track to the midi file
        track.append(mido.MetaMessage('track_name', name=rec+' '+str(pent[j]))) # Set the name of the midi track to the filename + the pitch
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(150), time=0)) # Set the tempo in beats per minute
        if j%2 == 0: # Change channels and controls every two filtered EQ frequencies
            chan = [0,1, 9,3,4, 5,6,7,8, 9,9,9, 13,14,15,15][int(j/2)] # There are 16 midi channels but 10 (index 9) is only percussion
            #inst = [66,66, 56,56,56, 57,57,57,57, 71,71,71, 75,75,75,75][int(j/2)] # Set instruments to use
            inst = [34,34, 35,66,66, 34,34,66,66, 35,56,42, 75,75,75,75][int(j/2)]
            random.seed(0)
            pan = random.sample(range(4,128,8),16)[int(j/2)] # Generate random pan positions (min possible = 0, max possible = 127)
        track.append(mido.Message('program_change', channel=chan, program=inst)) # Set instrument for the channel
        track.append(mido.Message('control_change', channel=chan, control=10, value=pan)) # Set pan for the channel
        for k in np.arange(len(avg[j])):
            curr_avg = avg[j][k] # Select the current value of the averaged accelerations
            if curr_avg < base: # If the acceleration is below the allowed range make it silent
                vel = 0
                time_len = min_len
            elif curr_avg < top: # If the acceleration is within the allowed range interpolate normally
                vel = int(round(float(inter_vol(curr_avg))))
                time_len = int(round(float(inter_len(curr_avg))))
            else: # If the acceleration is above the allowed range assign it maximum values
                vel = max_vol
                time_len = max_len
                big = True
            delay = int(time_len*0.1) # Set a delay at the start of each block that is proportional to the note length
            track.append(mido.Message('note_on', channel=chan, note=pent[j], velocity=vel, time=delay)) # Turn note on after the delay
            track.append(mido.Message('note_off', channel=chan, note=pent[j], time=(time_len-delay))) # Turn note off after the remaining duration
            ticks = time_len
            while ticks < (max_len - 2*time_len): # Play notes to fill the time block
                track.append(mido.Message('note_on', channel=chan, note=pent[j], velocity=vel, time=time_len)) # Turn note on after 1 duration
                track.append(mido.Message('note_off', channel=chan, note=pent[j], time=time_len)) # Turn note off after 1 duration
                ticks += 2*time_len
            track.append(mido.Message('note_off', channel=chan, note=pent[j], time=(max_len-ticks))) # Leave the note off for the rest of the block
    if big: # If any average acceleration in the recording was above the interpolation range
        print("\nThe maximum acceleration averaged over {0} seconds exceeded {1} cm/s2 \n{2} underrepresents the ground acceleration's intensity".format(str(block),str(top),rec))
        rec = rec+' (over max)' # Change the file name to indicate it went over the max acceleration 
    midi_file.save(rec+'.mid') # Save the midi file for the recording
    print('\nCompleted MIDI file for {0}'.format(rec))

    # Convert the midi file to audio (.flac format) using FluidSynth
    if platform.system()=='Darwin': # Mac sometimes prints two fluidsynth: panic messages, not sure why, the audio comes out fine
        print('\nIgnore the following two "fluidsynth: panic" messages')
    subprocess.run(['fluidsynth', '-o','synth.reverb.width=80', '-o','synth.reverb.room-size=0.8',
                    '-ni', '-q', '-g','0.5', 'MuseScore_General.sf2', rec+'.mid','-F',rec+'.flac'])
    print('\nSaved audio as a flac file for {0}'.format(rec))

def animate_cursor(dt, vals, avg, file, title):
    """ Animate a time cursor moving across plots of the clipped ground acceleration and averaged accelerations """
    print('\nBeginning animation for {0}'.format(file))
    # Set up the figure for the two plots
    fig = plt.figure(num=4, figsize=(12.8,8))
    gs = fig.add_gridspec(nrows=2, hspace=0, wspace=0, height_ratios=[1,2])
    axs = gs.subplots(sharex=True)
    fig.suptitle(file+'\n'+title)
    props = dict(facecolor='white', alpha=1) # Set the properties for the text boxes

    # Create a copy of the clipped ground acceleration plot to animate the cursor on
    x1 = np.linspace(0, dt*len(vals), len(vals))
    axs[0].plot(x1, vals, label='Clipped Ground Acceleration', color='tab:blue')
    axs[0].set_ylabel(r'Acceleration [cm/s$^2$]')
    axs[0].set_xlim(0, dt*len(vals))
    mult = 1.2 # Set the multiplier to decide y-axis limits
    axs[0].set_ylim(mult*np.min(vals), mult*np.max(vals))
    axs[0].text(0.38, 0.9, 'Clipped Ground Acceleration Recording', transform=axs[0].transAxes, bbox=props)

    # Create a copy of the average acceleration plot to animate the cursor on
    x2 = np.arange(block/2, block*len(avg[0]), block)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.inferno(np.linspace(0,1,len(avg))))
    for i in np.arange(len(avg)):
        axs[1].plot(x2, avg[i], label='Average Acceleration', alpha=0.6)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_xlim(0, block*len(avg[0]))
    axs[1].set_ylabel(r'Mean Acceleration [cm/s$^2$]')
    axs[1].set_ylim(base,top)
    axs[1].text(0.42, 0.95, 'Mean Filtered Acceleration', transform=axs[1].transAxes, bbox=props)

    # Create a gradient colorbar as a legend
    cbar_ax = axs[1].inset_axes([0.985,0.125,0.015,0.77])
    for i in np.arange(len(avg)):
        cbar_ax.barh(i, width=1, height=1, align='edge', linewidth=0, alpha=0.6)
    cbar_ax.set_ylim(0,len(avg))
    cbar_ax.set_yticks([0,8,16,24,32])
    cbar_ax.set_yticklabels(['0','0.5','3','17','100'])
    cbar_ax.set_ylabel('Wave Frequency [Hz]')
    cbar_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Apply a tight layout with ticks only on the outer axes
    axs[0].label_outer()
    axs[1].label_outer()
    fig.tight_layout()

    # Animate the motion of the cursor
    start = ani_start # Start time
    end = start + ani_dur # End time
    if end > dt*len(vals): # If the end time is greater than the audio duration set end = audio duration
        end = dt*len(vals)
        if start > end: # If the start time is greater than the end time set start = 0 seconds
            start = end - ani_dur
    c_x = np.arange(start, end+ani_step/2, ani_step) # Set up the array of cursor positions
    cursor1 = axs[0].plot([c_x[0],c_x[0]], [mult*np.min(vals), mult*np.max(vals)], color='k') # Plot the cursor on plot 1
    cursor2 = axs[1].plot([c_x[0],c_x[0]], [base,top], color='k') # Plot the cursor on plot 2
    def animate(i): # Create the function that will make the cursor step through the c_x array
        plt.setp(cursor1, data=([i,i], [mult*np.min(vals), mult*np.max(vals)]))
        plt.setp(cursor2, data=([i,i], [base,top]))
        return (cursor1,cursor2)
    ani = animation.FuncAnimation(fig, animate, frames=c_x, interval=int(ani_step*1000)) # Animate the cursor
    vid_writer = animation.FFMpegWriter(fps=1/ani_step, bitrate=500) # Set the export quality of the file
    ani.save('{0} {1} ({2:.1f} to {3:.1f}).mp4'.format(file,'animation',start,end), writer=vid_writer) # Save the animation as a .mp4
    plt.close()
    print('\nCompleted animation and saved mp4 file for {0} {1}'.format(file,'animation'))


# Set key processing variables
base = 0.1 # [cm/s^2] Set the minimum acceleration that counts as an earthquake
# The minimum average acceleration for the interpolation range is the base value
top = 30 # [cm/s^2] Set the maximum average acceleration for the interpolation range
block = 0.5 # [s] Number of seconds in each time block
bands = create_bands() # Generate the centre frequencies for bandpass filtering
pent = create_pent(root=34) # Generate a pentatonic scale starting on MIDI pitch 34 (Bb1)

# Format earthquakes array and create dictionaries for results
earthquakes = []
for i in np.arange(len(events)):
    earthquakes.append([events[i]+'/'+recordings[i][n] for n in np.arange(len(recordings[i]))])
waves = {} # Create the dictionary to deposit filtered waves in
averages = {} # Create the dictionary to deposit averaged accelerations in

# Process the data for each earthquake
for eq in earthquakes:
    # Extract the ground acceleration data for each earthquake using the sample code from seisfinder
    dts, values = sample_extraction(eq)
    if in_g==True: # If required convert from units of gravity to units of cm/s2
        for i in np.arange(len(eq)):
            values[i] = values[i]*981
    
    # Plot the recorded ground acceleration at all recording sites in a single figure
    plot_recordings(eq)
    
    # Process each recording
    for i in np.arange(len(eq)):
        # If the ground acceleration data is all smaller than the base value abort the conversion
        if np.all(values[i]<base):
            print('\nNo data point in {0} was above {1} cm/s2 so the ground acceleration was not converted'.format(eq[i], base))
            continue

        # Filter the data and store in 'waves'
        (filtered_vals, filtered_bands, filtered_sum, clipped) = filter(dts[i], values[i], eq[i])
        # Plot the filtered frequencies and compare to the original ground acceleration
        plot_filtered(dts[i], clipped, filtered_vals, filtered_sum, filtered_bands, eq[i], 'Comparison of Filtered Approximation to Clipped Ground Acceleration Recording')

        # Number of data points in each time block for the recording
        span = int(block/dts[i])
        # Average the accelerations for each frequency over each time block and store in 'averages'
        average_amp(waves[eq[i]], eq[i], span)
        # Plot the average accelerations for all frequencies in the recording
        plot_avg(averages[eq[i]], filtered_bands, eq[i], 'Mean Filtered Acceleration')

        # Create and save a .mid MIDI file and .flac audio file for the filtered ground accelerations
        audio(averages[eq[i]], eq[i])
        # Animate a time cursor moving across plots of the clipped ground acceleration and averaged accelerations
        if make_ani == True:
            animate_cursor(dts[i], clipped, averages[eq[i]], eq[i], 'Visualisation of Audio')

print('\nProcessing complete\n')