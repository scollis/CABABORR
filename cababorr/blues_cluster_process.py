#!/usr/bin/env python
"""
cababorr.blues_cluster_process
=========================
top level script for processing a cpol file from command line
on Argonne LCRC Blues
.. autosummary::
    :toctree: generated/
    hello_world
"""
from matplotlib import use
use('agg')
#import processing_code as radar_codes
import sys
from glob import glob
import os, platform
import pyart
import netCDF4
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy import ndimage, signal, integrate
import time
from time import sleep
import copy
import netCDF4
#import skfuzzy as fuzz
import datetime
import fnmatch
from ipyparallel import Client

def get_file_tree(start_dir, pattern):
    """
    Make a list of all files matching pattern
    above start_dir
    Parameters
    ----------
    start_dir : string
        base_directory
    pattern : string
        pattern to match. Use * for wildcard
    Returns
    -------
    files : list
        list of strings
    """

    files = []

    for dir, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir, pattern)))
    return files

def process_a_volume(packed):
    from matplotlib import use
    use('agg')
    import os
    import sys
    import glob
    import time
    import logging
    import argparse
    import datetime
    import warnings
    import matplotlib
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from copy import deepcopy
    import operator
    import pyart
    import netCDF4
    import numpy as np
    import pytz
    import cartopy
    import platform
    import scipy

    # Custom modules.
    import imp
    lib_loc = os.path.join(os.path.expanduser('~'), 'projects/CABABORR/cababorr/processing_code.py')

    radar_codes = imp.load_source('radar_codes', lib_loc)

    soundings_dir = packed['s_dir']
    odir_radars = packed['odir_r']
    odir_statistics = packed['odir_s']
    odir_images = packed['odir_i']
    radar_fname = packed['infile']


    radar = pyart.io.read(radar_fname)
    radar_start_date = netCDF4.num2date(radar.time['data'][0],
            radar.time['units'])
    print(radar_start_date)
    ymd_string = datetime.datetime.strftime(radar_start_date, '%Y%m%d')
    hms_string = datetime.datetime.strftime(radar_start_date, '%H%M%S')
    lats = radar.gate_latitude
    lons = radar.gate_longitude

    im_output_location = os.path.join(odir_images,ymd_string)
    if not os.path.exists(im_output_location):
        try:
            os.makedirs(im_output_location)
        except:
            print('looks like it is there! sneaky')

    z_dict, temp_dict, snr = radar_codes.snr_and_sounding_interp_sonde(radar,
            soundings_dir)
    texture =  radar_codes.get_texture(radar)

    radar.add_field('sounding_temperature', temp_dict, replace_existing = True)
    radar.add_field('height', z_dict, replace_existing = True)
    radar.add_field('SNR', snr, replace_existing = True)
    radar.add_field('velocity_texture', texture, replace_existing = True)
    blanker = deepcopy(radar.fields['cross_correlation_ratio'])
    blanker['data'] = np.ones(blanker['data'].shape)
    radar.add_field('normalized_coherent_power',
            blanker, replace_existing=True)

    my_fuzz, cats = radar_codes.do_my_fuzz(radar)
    radar.add_field('gate_id', my_fuzz,
                      replace_existing = True)

    cat_dict = {}
    for pair_str in radar.fields['gate_id']['notes'].split(','):
        print(pair_str)
        cat_dict.update({pair_str.split(':')[1]:int(pair_str.split(':')[0])})

    sorted_cats = sorted(cat_dict.items(), key=operator.itemgetter(1))


    cat_colors = {'rain' : 'green',
                'multi_trip' : 'red',
                'no_scatter' : 'gray',
                'snow' : 'cyan',
                'melting' : 'yellow'}

    min_lon = lons['data'].min()
    min_lat = lats['data'].min()
    max_lat = lats['data'].max()
    max_lon = lons['data'].max()

    lal = np.arange(min_lat, max_lat, .5)
    lat_lines = np.arange(min_lon, max_lon, .5)

    sw = 2

    display = pyart.graph.RadarMapDisplay(radar)

    f = plt.figure(figsize = [15,10])
    plt.subplot(2, 2, 1)
    lab_colors=['red','cyan', 'gray', 'green', 'yellow']
    lab_colors = [cat_colors[kitty[0]] for kitty in sorted_cats]

    cmap = matplotlib.colors.ListedColormap(lab_colors)
    display.plot_ppi_map('gate_id', sweep = sw,
                         min_lon = min_lon, max_lon = max_lon,
                         min_lat = min_lat, max_lat = max_lat,
                         resolution = 'l', cmap = cmap, vmin = 0, vmax = 5)
    cbax=plt.gca()
    #labels = [item.get_text() for item in cbax.get_xticklabels()]
    #my_display.cbs[-1].ax.set_yticklabels(cats)
    tick_locs   = np.linspace(0,len(cats) -1 ,len(cats))+0.5
    display.cbs[-1].locator     = matplotlib.ticker.FixedLocator(tick_locs)
    catty_list = [sorted_cats[i][0] for i in range(len(sorted_cats))]
    display.cbs[-1].formatter   = matplotlib.ticker.FixedFormatter(catty_list)
    display.cbs[-1].update_ticks()
    plt.subplot(2, 2, 2)
    display.plot_ppi_map('reflectivity', sweep = sw, vmin = -8, vmax = 64,
                          min_lon = min_lon, max_lon = max_lon,
                          min_lat = min_lat, max_lat = max_lat,
                         resolution = 'l', cmap = pyart.graph.cm.NWSRef)

    plt.subplot(2, 2, 3)
    display.plot_ppi_map('velocity_texture', sweep = sw, vmin =0, vmax = 14,
                         min_lon = min_lon, max_lon = max_lon,
                         min_lat = min_lat, max_lat = max_lat,
                         resolution = 'l', cmap = pyart.graph.cm.NWSRef)
    plt.subplot(2, 2, 4)
    display.plot_ppi_map('cross_correlation_ratio', sweep = sw,
                        vmin = .5, vmax = 1,
                          min_lon = min_lon, max_lon = max_lon,
                          min_lat = min_lat, max_lat = max_lat,
                         resolution = 'l', cmap = pyart.graph.cm.Carbone42)

    plt.savefig(os.path.join(im_output_location,
        'multi_'+ymd_string+hms_string+'.png'))

    happy_gates = pyart.correct.GateFilter(radar)
    happy_gates.exclude_all()
    happy_gates.include_equal('gate_id', cat_dict['rain'])
    happy_gates.include_equal('gate_id', cat_dict['melting'])
    happy_gates.include_equal('gate_id', cat_dict['snow'])
    melt_locations = np.where(radar.fields['gate_id']['data'] == 1)
    kinda_cold = np.where(radar.fields['sounding_temperature']['data'] < 0)
    fzl_sounding = radar.gate_altitude['data'][kinda_cold].min()
    if len(melt_locations[0] > 1):
        fzl_pid = radar.gate_altitude['data'][melt_locations].min()
        fzl = (fzl_pid + fzl_sounding)/2.0
    else:
        fzl = fzl_sounding

    print(fzl)
    if fzl > 5000:
        fzl = 3500.0

    phidp, kdp = pyart.correct.phase_proc_lp(radar,
            0.0, debug=True, fzl=fzl)
    radar.add_field('corrected_differential_phase',
            phidp,replace_existing = True)
    radar.add_field('corrected_specific_diff_phase',
            kdp,replace_existing = True)
    csu_kdp_field, csu_filt_dp, csu_kdp_sd = radar_codes.return_csu_kdp(radar)
    radar.add_field('bringi_differential_phase',
            csu_filt_dp, replace_existing = True)
    radar.add_field('bringi_specific_diff_phase',
            csu_kdp_field, replace_existing = True)
    radar.add_field('bringi_specific_diff_phase_sd',
            csu_kdp_sd, replace_existing = True)
    try:
        m_kdp, phidp_f, phidp_r = pyart.retrieve.kdp_proc.kdp_maesaka(radar,
                                gatefilter=happy_gates)
    except IndexError:
        m_kdp = deepcopy(csu_kdp_field)
        m_kdp['data'] = m_kdp['data'] * 0.0
        phidp_f = deepcopy(csu_filt_dp)
        phidp_f['data'] = phidp_f['data'] * 0.0
        phidp_r = phidp_f

    radar.add_field('maesaka_differential_phase', m_kdp,
            replace_existing = True)
    radar.add_field('maesaka_forward_specific_diff_phase',
            phidp_f, replace_existing = True)
    radar.add_field('maesaka__reverse_specific_diff_phase',
            phidp_r, replace_existing = True)

    display = pyart.graph.RadarMapDisplay(radar)
    fig = plt.figure(figsize = [20,6])
    plt.subplot(1,3,1)
    display.plot_ppi_map('bringi_specific_diff_phase', sweep = 0, resolution = 'l',
                        mask_outside = False,
                        cmap = pyart.graph.cm.NWSRef,
                        vmin = 0, vmax = 6, title='Bringi/CSU',
                        gatefilter=happy_gates)
    plt.subplot(1,3,2)
    display.plot_ppi_map('corrected_specific_diff_phase', sweep = 0, resolution = 'l',
                        mask_outside = False,
                        cmap = pyart.graph.cm.NWSRef,
                        vmin = 0, vmax = 6, title='Giangrande/LP',
                        gatefilter=happy_gates)

    plt.subplot(1,3,3)
    display.plot_ppi_map('maesaka_differential_phase', sweep = 0, resolution = 'l',
                        mask_outside = False,
                        cmap = pyart.graph.cm.NWSRef,
                        vmin = 0, vmax = 6, title='North/Maesaka',
                        gatefilter=happy_gates)

    plt.savefig(os.path.join(im_output_location,
        'csapr_kdp_comp_'+ymd_string+hms_string+'.png'))

    height = radar.gate_altitude
    lats = radar.gate_latitude
    lons = radar.gate_longitude
    lowest_lats = lats['data'][radar.sweep_start_ray_index['data'][0]:radar.sweep_end_ray_index['data'][0],:]
    lowest_lons = lons['data'][radar.sweep_start_ray_index['data'][0]:radar.sweep_end_ray_index['data'][0],:]
    c1_dis_lat = -12.4389
    c1_dis_lon = 130.9556
    cost = np.sqrt((lowest_lons - c1_dis_lon)**2 + (lowest_lats - c1_dis_lat)**2)
    index = np.where(cost == cost.min())
    lon_locn = lowest_lons[index]
    lat_locn = lowest_lats[index]
    print(lat_locn, lon_locn)
    dis_output_location = os.path.join(odir_statistics, ymd_string)
    if not os.path.exists(dis_output_location):
        try:
            os.makedirs(dis_output_location)
        except:
            print('looks like it is there! sneaky')

    dis_string = ''
    time_of_dis = netCDF4.num2date(radar.time['data'], radar.time['units'])[index[0]][0]
    tstring = datetime.datetime.strftime(time_of_dis, '%Y%m%d%H%H%S')
    dis_string = dis_string + tstring + ' '
    for key in radar.fields.keys():
        dis_string = dis_string + key + ' '
        dis_string = dis_string + str(radar.fields[key]['data'][index][0]) + ' '


    write_dis_filename = os.path.join(dis_output_location,
                                     'csapr_distro_'+ymd_string+hms_string+'.txt')
    dis_fh = open(write_dis_filename, 'w')
    dis_fh.write(dis_string)
    dis_fh.close()
    hts = np.linspace(radar.altitude['data'],15000.0 + radar.altitude['data'],61)
    flds =['reflectivity',
         'bringi_specific_diff_phase',
         'corrected_specific_diff_phase',
         'maesaka_differential_phase',
         'cross_correlation_ratio',
         'velocity_texture']
    my_qvp = radar_codes.retrieve_qvp(radar, hts, flds = flds)
    hts_string = 'height(m) '
    for htss in hts:
        hts_string = hts_string + str(int(htss)) + ' '

    write_qvp_filename = os.path.join(dis_output_location,
                                     'csapr_qvp_'+ymd_string+hms_string+'.txt')


    dis_fh = open(write_qvp_filename, 'w')
    dis_fh.write(hts_string + '\n')
    for key in flds:
        print(key)
        this_str = key + ' '
        for i in range(len(hts)):
            this_str = this_str + str(my_qvp[key][i]) + ' '
        this_str = this_str + '\n'
        dis_fh.write(this_str)
    dis_fh.close()

    r_output_location = os.path.join(odir_radars, ymd_string)
    if not os.path.exists(r_output_location):
        try:
            os.makedirs(r_output_location)
        except:
            print('looks like it is there! sneaky')

    rfilename = os.path.join(r_output_location, 'csaprsur_' + ymd_string + '.' +  hms_string + '.nc')
    pyart.io.write_cfradial(rfilename, radar)
    return None



if __name__ == "__main__":
    if len(sys.argv) > 1:
        patt = sys.argv[1]
    else:
        patt = None

    my_system = platform.system()
    #hello_world()
    if my_system == 'Darwin':
        top = '/data/sample_sapr_data/sgpstage/sur/'
        s_dir = '/data/sample_sapr_data/sgpstage/interp_sonde/'
        odir_r = '/data/sample_sapr_data/agu2016/radars/'
        odir_s = '/data/sample_sapr_data/agu2016/stats/'
        odir_i = '/data/sample_sapr_data/agu2016/images/'
    elif my_system == 'Linux':
        top = '/lcrc/group/earthscience/radar/stage/radar_disk_two/cpol_lassen/cpol_0506/'
        s_dir = '/lcrc/group/earthscience/radar/stage/interpsonde/'
        odir_r = '/lcrc/group/earthscience/radar/egu17/radars/'
        odir_s = '/lcrc/group/earthscience/radar/egu17/stats/'
        odir_i = '/lcrc/group/earthscience/radar/egu17/images/'

    all_files = get_file_tree(top, '*.lassen')
    print('found ', len(all_files), ' files')
    good_files = []
    for ffile in all_files:
        if 'PPI' in ffile:
            statinfo = os.stat(ffile)
            if statinfo.st_size > 10*1e6:
                good_files.append(ffile)

    if patt is None:
        really_good = good_files
    else:
        really_good = []
        for ffile in good_files:
            if patt in ffile:
                really_good.append(ffile)

    print(len(good_files))
    print(len(really_good))

    packing = []
    for fn in really_good:
        this_rec = {'top' : top,
                's_dir' : s_dir,
                'odir_r' : odir_r,
                'odir_s' : odir_s,
                'odir_i' : odir_i,
                'infile' : os.path.join(top, fn)}
        packing.append(this_rec)

    good = False
    while not good:
        try:
            My_Cluster = Client()
            My_View = My_Cluster[:]
            print(My_View)
            print(len(My_View))
            good = True
        except:
            print('no!')
            sleep(15)
            good = False

    My_View.block = False
    result = My_View.map_async(process_a_volume, packing)
    #result = My_View.map_async(test_script, packing[0:100])

    #Reduce the result to get a list of output
    qvps = result.get()
    print(qvps)








