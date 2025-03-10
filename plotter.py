from os import system
import sys
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


UBUCHAN_PEAK_BANDWIDTH = 672.3 * 1e9 #GigaBytes
UBUCHAN_PEAK_FLOPS = 44.10 * 1e12 #TeraFlops 

colors = {"gern": "blue"}

def roofline_data(peak_bandwidth, peak_flops, ax):
    # Compute the cutoff point
    cutoff = (peak_flops/peak_bandwidth) * 4
    print(cutoff)
    #generate the operational intensity
    point1 = (0, 0) 
    point2 = (cutoff, peak_flops)
    # make the memory bound portion of the graph
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')

    # plt.axvline(x=cutoff, color='r', linestyle='--', linewidth=2, label=f'{cutoff}')

    x_axis_limit = ax.get_xlim()[1]
    point1 = (cutoff, peak_flops) 
    point2 = (x_axis_limit, peak_flops)
    #make the compute portion of the graph
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-', label="Roofline")
    return ax


def create_figure(x_lim):
    fig, ax = plt.subplots()
    ax.set_xlabel('Operational Intensity')
    ax.set_ylabel('Attainable FLOPS/s')
    ax.set_title('GPU Roofline')
    ax.set_xlim(left=0, right=x_lim)
    ax.legend()
    return fig, ax

def read_result(filename):
    x = []
    y = []
    
    data = open(filename)
    lines = data.readlines() 

    for i in range(0, len(lines), 2):
        x.append(float(lines[i]))
        y.append(float(lines[i+1][:-1]))    

    return x, y 

def make_df(systems, directory): 
    data = None
    for system in systems:
        x,y = read_result(f'{directory}/{system}')
        df = pd.DataFrame({'xaxis' : x, system : y})
        if data is None:
            data = df
        else:
            data = pd.merge(data, df, on='xaxis', how='outer')

    return data 

def do_nothing(x):
    return x

def generate_plot(systems, dir, title, x_label, y_label, name,
                  legend, colors, 
                  apply_x=do_nothing, apply_y=do_nothing, 
                  roofline=True, bandwidth=False,
                  log_x=False, log_y=False):

    # assert roofline ^ bandwidth
    
    df = make_df(systems, dir)
    df['xaxis'] = apply_x(df['xaxis'])
    print(df)

    fig, ax = plt.subplots()
    # print(df)
    for i in range(len(systems)): 
        ax.plot(df['xaxis'], df[systems[i]], label=legend[i], color=colors[i])

    plt.rcParams.update({
    'font.size': 20,          # Default text size
    'axes.titlesize': 16,     # Title size
    'axes.labelsize': 20,     # Axes labels size
    'xtick.labelsize': 16,    # X-axis tick labels size
    'ytick.labelsize': 16,    # Y-axis tick labels size
    'legend.fontsize': 16,    # Legend font size
    'figure.titlesize': 16    # Figure title size
    })

    if roofline: 
        roofline_data(UBUCHAN_PEAK_BANDWIDTH, UBUCHAN_PEAK_FLOPS, ax)
    # else:
    #     plt.axhline(y=UBUCHAN_PEAK_FLOPS, color='r', linestyle='--', linewidth=2, label='44 TFLOP/s (Peak) ')

    if bandwidth: 
        plt.axhline(y=UBUCHAN_PEAK_BANDWIDTH, color='r', linestyle='--', linewidth=2, label='672 Gb/s (Peak Bandwidth)') 

    if log_y:
        plt.yscale("log")
      

    if log_x:
        plt.xscale("log")

    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(frameon=False, fontsize=16)

    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_title(title)
    
    # plt.savefig(f'{dir}/{name}.pdf', format="pdf", bbox_inches='tight')
    plt.savefig(f'{dir}/{name}.png', format="png", bbox_inches='tight')
    # plt.savefig(f'{dir}/{name}_raw_graph.svg', format="svg", bbox_inches='tight')

def get_speedup_up(systems, compare_to, dir, 
                  title, x_label, y_label, name,
                  legend, colors, 
                  apply_x=do_nothing, apply_y=do_nothing, flops=False):

    df = make_df(systems, dir)
    df['xaxis'] = apply_x(df['xaxis'])
    fig, ax = plt.subplots()

    for system in systems:
        if flops:
            df[f"{system}_relative"] = df[system]/df[compare_to]
        else:
          df[f"{system}_relative"] = df[compare_to]/df[system]

    print(df)
    
    for i in range(len(systems)): 
        ax.plot(df['xaxis'], df[f"{systems[i]}_relative"], label=legend[i], color=colors[i])

    # ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend(frameon=False, fontsize=16)

    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_title(title)
    plt.savefig(f'{dir}/{name}.png', format="png", bbox_inches='tight')



    
# def num_bytes_blur(x):
#     return x * x * 4 * 2 # * 2 for two inputs

generate_plot(["gern", "triton"], 
               "apps/blur",
              "Blur X -> Blur Y", "Size of (N X N) Image", 
              "MB/s (Bandwidth)", "bandwidth_gern", 
              ['Gern', 'Triton'], 
              [colors["gern"], "green"],
               roofline = False,
               bandwidth = True)
