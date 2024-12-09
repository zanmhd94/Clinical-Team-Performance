import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import pandas as pd


def array_func(a=None, f=None, args_pos=None, args_dict=None, reshape=None):


    if f is None:
        return a 
    else:
        if args_pos is None:
            args_pos = ()
        if args_dict is None:
            args_dict = {}

            
        if reshape is None:
            reshape = True

        if a is None:
            a = []
        a = np.array(a)
        a_shape = np.array(np.shape(a))

        N_i = np.product(a_shape)

        a = np.reshape(a, (N_i))

        r = np.array([f(a[i], *args_pos, **args_dict) for i in range(N_i)])

        if reshape:
            return np.reshape(r, a_shape)
        else:
            return r
        

def is_leapyear(years):

    return np.logical_and((years%4)==0, (years%100)!=0)

def is_leapday(s):

    return s[5:10]=="02-29"


def invalid_leapday(s):

    is_ly = is_leapyear(np.array(np.array(s, dtype='<U4'), dtype=int))
    is_ld = array_func(a=np.array(s, dtype='<U10'), f=is_leapday)

    return np.logical_and(is_ld, ~is_ly)


def timedelta_mean(X):

    if np.size(X)==0:
        return pd.Timedelta("NaT")
    else:
        return np.mean(X)

def timedelta_std(X):

    if np.size(X)==0:
        return pd.Timedelta("NaT")
    else:
        return np.std(X)

def timedelta_quantile(X, q=None):
    if q is None:
        q = 0.5
    if np.size(X)==0:
        return pd.Timedelta("NaT")
    else:
        return np.quantile(X, q)


def make_iterable_array(values, shape=None, dtype=None, as_list=None):

    if as_list is None:
        as_list = False

    if np.isin([type(values)], [np.ndarray, np.matrix]) == False:
        values = np.array(values)
    if type(shape) == type(None):
        shape = values.size
    if np.product(shape) != values.size:
            raise ValueError("Mismatch between size of values and specified output shape")

    if dtype is not None:
        values = np.array(np.resize(values, shape), dtype=dtype)
    else:
        values = np.resize(values, shape)

    if as_list:
        return [v for v in values]
    else:
        return values


def get_index(df, colval=None):
   
   if colval is None:
       return pd.Series(np.repeat(True, df.shape[0]), index=df.index)
   else:
        return pd.Series(np.all(np.array([np.any(np.array([df[col].values == val for val in make_iterable_array(colval[col])]), axis=0) for col in colval]), axis=0), index=df.index)


def index_df(df, colval=None):
    if colval is None:
        return df
    else:
        return df[get_index(df=df, colval=colval)]
    


def mkdir_export(path, mkdir=None):

    if mkdir is None:
        mkdir = True
    if mkdir:
        if os.path.exists(path)==False:
            os.mkdir(path)
    return path


class rc_fig():

    def __init__(self, name=None, nrows=None, ncols=None, wratio=None, hratio=None, borderlabels=None, tick_direction=None):

        self.rc_plot_setup(name=name, nrows=nrows, ncols=ncols, wratio=wratio, hratio=hratio, borderlabels=borderlabels, tick_direction=tick_direction)

    def rc_plot_setup(self, name=None, nrows=None, ncols=None, wratio=None, hratio=None, borderlabels=None, tick_direction=None):

        if name is None:
            name = "fig"

        hratio = make_iterable_array(hratio)
        wratio = make_iterable_array(wratio)


        if hratio[0] is not None:
            nrows = np.size(hratio)
        else:
            if nrows is None:
                nrows = 1
            hratio = np.ones(nrows, dtype=int)

        if wratio[0] is not None:
            ncols = np.size(wratio)
        else:
            if ncols is None:
                ncols = 1
            wratio = np.ones(ncols, dtype=int)

        hratio = make_iterable_array(hratio, dtype=int)
        wratio = make_iterable_array(wratio, dtype=int)

        cs_hratio = np.concatenate([[0], np.cumsum(hratio)])
        cs_wratio = np.concatenate([[0], np.cumsum(wratio)])

        if borderlabels is None:
            borderlabels = 'leftbottom'


        if tick_direction is None:
            tick_direction = "out"

        
        self.name = name
        self.fig = plt.figure(constrained_layout=True)
        self.gs = self.fig.add_gridspec(cs_hratio[-1], cs_wratio[-1])
        self.ax = [[None for j in range(ncols)] for i in range(nrows)]

        for i in range(len(self.ax)):
            for j in range(len(self.ax[i])):
                self.ax[i][j] = self.fig.add_subplot(self.gs[cs_hratio[i]:cs_hratio[1+i], cs_wratio[j]:cs_wratio[1+j]])


        for i in range(len(self.ax)):
            for j in range(len(self.ax[i])):
                self.ax[i][j].tick_params(
                                                which='both',      
                                                bottom=True, labelbottom=False,     
                                                top=True, labeltop=False,
                                                left=True, labelleft=False,     
                                                right=True, labelright=False,         
                                                direction=tick_direction,
                                            )

        if borderlabels=='edge':
            for i in range(len(self.ax)):
                for j in range(len(self.ax[i])):
                    if i==0:
                        if len(self.ax) > 1:
                            self.ax[i][j].xaxis.set_label_position("top")
                            self.ax[i][j].tick_params(labeltop=True)
                    if i==len(self.ax)-1:
                        self.ax[i][j].tick_params(labelbottom=True)
                if len(self.ax[i]) > 1:
                    self.ax[i][-1].yaxis.set_label_position("right")
                    self.ax[i][-1].tick_params(labelright=True)
                self.ax[i][0].tick_params(labelleft=True)
        elif borderlabels=='leftbottom':
            for i in range(len(self.ax)):
                for j in range(len(self.ax[i])):
                    if i==len(self.ax)-1:
                        self.ax[i][j].tick_params(labelbottom=True)
                self.ax[i][0].tick_params(labelleft=True)
        elif borderlabels=='leftbottomedge':
            for i in range(len(self.ax)):
                for j in range(len(self.ax[i])):
                    if i==len(self.ax)-1:
                        self.ax[i][j].tick_params(labelbottom=True)
                    self.ax[i][j].tick_params(labelleft=True)
        elif borderlabels=='leftedgebottomedge':
            for i in range(len(self.ax)):
                for j in range(len(self.ax[i])):
                    self.ax[i][j].tick_params(labelbottom=True)
                self.ax[i][0].tick_params(labelleft=True)
        else:
            for i in range(len(self.ax)):
                for j in range(len(self.ax[i])):
                    self.ax[i][j].tick_params(labelleft=True, labelbottom=True)


        self.fig.set_size_inches(7, 4)
        self.fig.tight_layout()

    def twin(self, axis=None):

        if axis is None:
            axis = "y"

        if axis=="x":
            self.ax2 = [[self.ax[i][j].twinx() for j in range(len(self.ax[i]))] for i in range(len(self.ax))]
        else:
            self.ax2 = [[self.ax[i][j].twiny() for j in range(len(self.ax[i]))] for i in range(len(self.ax))]


    def export(self, fig_dir=None, formats=None):

        if formats is None:
            formats = ["pdf", "svg"]
        formats = make_iterable_array(formats)

        if fig_dir is None:
            fig_dir = "."
        else:
            mkdir_export(fig_dir)

        self.fig.tight_layout()
        for fm in formats:
            self.fig.savefig(f"{fig_dir}/{self.name}.{fm}", format=fm)


