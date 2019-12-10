import maxflow
import numpy as np
import librosa as lr
from librosa import display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
# from pathos.multiprocessing import ProcessingPool as Pool
# import pyrubberband
import numba
from numba import jit

class TFC:
    """
    This class controls all of the input, output, and processing to calculate and complete a Time-Frequency Crossfade.
    """

    def __init__(self, sr=None, n_fft=2048):
        self.parameters = {
            'sr': sr,
            'n_fft': n_fft
        }

    def load_audio(self, file, trim=True):
        """
        Loads audio from the directory given in file and returns the raw data. The sample rate used is either the
        sample rate set manually or the native sample rate of the audio if not specified.
        """
        y, sr = lr.core.load(file, sr=self.parameters['sr'])

        if not self.parameters['sr']:
            self.parameters['sr'] = sr

        if trim:
            yt, i = lr.effects.trim(y)
            return yt
        else:
            return y

    def write_audio(self, file, y):
        """
        Saves the audio data the the specified file location as a wav file.
        :param file:
        :param song:
        :param sr:
        :return:
        """
        lr.output.write_wav(file, y, sr=self.parameters['sr'])

    def stft(self, y, n_fft=2048):
        """
        Returns a discretized representation of the audio consisting of a series of complex values using a short-term
        fourier transform.
        :param y:
        :param n_fft:
        :return:
        """
        return lr.stft(y, n_fft=n_fft)

    def istft(self, yft):
        """
        Reconstructs the audio from provided stft transformed data.
        :param yft:
        :return:
        """
        return lr.core.istft(yft)

    def get_overlap(self, y1, y2, seconds):
        """
        Computes and returns the transition length in samples.
        :param y1:
        :param y2:
        :param seconds:
        :return:
        """
        samples = round(self.parameters['sr']*seconds)

        return samples

    def build_graph(self, y1ft, y2ft, loss=None):
        """
        Builds and returns a flow-graph on adjacent time-freqency bins using the provided loss function
        :param y1ft:
        :param y2ft:
        :param loss:
        :return:
        """

        # def simple_loss(a1, a2, b1, b2):
        #    return np.linalg.norm([a1 - b1]) + np.linalg.norm([a2 - b2])

        # if not loss:
        #    loss = simple_loss

        graph = maxflow.Graph[float]()
        node_ids = graph.add_grid_nodes((y1ft.shape[0], y1ft.shape[1]))

        @jit(nopython=True)
        def compute_weights(node_ids, y1ft, y2ft):
            edges = []

            ceil = y1ft.max()

            for row in range(y1ft.shape[0]):
                for x in range(y1ft.shape[1]):
                    if row != y1ft.shape[0] - 1:
                        edges.append([
                            node_ids[row, x],
                            node_ids[row + 1, x],
                            abs(y1ft[row, x] - y2ft[row, x]) + abs(y1ft[row + 1, x] - y2ft[row + 1, x]),
                            0])

                    if x != y1ft.shape[1] - 1:
                        edges.append([
                            node_ids[row, x],
                            node_ids[row, x + 1],
                            abs(y1ft[row, x] - y2ft[row, x]) + abs(y1ft[row, x + 1] - y2ft[row, x + 1]),
                            0])
            return edges

        edges = compute_weights(node_ids, y1ft, y2ft)

        for row in range(y1ft.shape[0]):
            graph.add_tedge(node_ids[row, 0], 999999, 0)
            graph.add_tedge(node_ids[row, y1ft.shape[1] - 1], 0, 999999)

        for edge in edges:
            graph.add_edge(edge[0], edge[1], edge[2], edge[3])

        return graph, node_ids

    def cut(self, graph, node_ids):
        """
        Computes the optimal graph cut.
        :param graph:
        :return:
        """
        flow = graph.maxflow()
        print("Flow = "+str(flow))
        seam = graph.get_grid_segments(node_ids)
        return seam

    def join_on_seam(self, y1ft, y2ft, seam):
        """
        Joins two stft representations of songs with equal dimensions along a found seam.
        :param y1ft:
        :param y2ft:
        :param seam:
        :return:
        """
        new_slice = np.zeros((y1ft.shape[0], y1ft.shape[1]), dtype=np.complex64)

        for row in range(seam.shape[0]):
            for x in range(seam.shape[1]):
                if seam[row, x]:
                    new_slice[row, x] = np.array(y2ft[row, x])
                    # new_slice[row,x] = 999
                else:
                    new_slice[row, x] = np.array(y1ft[row, x])
                    # new_slice[row,x] = np.array(70)

        return new_slice

    def visualize_seam(self, trans, seam):
        """
        Returns a visualization of a computed seam given the transition audio and seam matrix. The stft of the audio
        must be the same dimensions as the seam.
        :param trans:
        :param seam:
        :return:
        """
        yft = lr.amplitude_to_db(np.abs(self.stft(trans)), ref=np.max)
        line_width = round(yft.shape[1] / 150)

        for ri, row in enumerate(seam):
            ci = next(i for i, v in enumerate(row) if v)
            for highlight in range(max(ci-line_width, 0), min(ci+line_width, len(row))):
                yft[ri][highlight] = 0

        #for ri, row in enumerate(seam):
        #    for ci, col in enumerate(row):
        #        if seam[ri, ci]:
        #            yft[ri, ci] = 0


        plt.figure(figsize=(10, 4))
        display.specshow(yft, y_axis='log', x_axis='time', hop_length=self.parameters['n_fft']/8)
        #plt.show()

        output = io.BytesIO()
        plt.savefig(output, format='svg')

        return output

    def process_TFC(self, file1, file2, seconds, trim=True, loss=None):
        """
        Given two audio files computes and returns the files transitioned along a computed seam through the time-frequency
        domain.
        :param file1:
        :param file2:
        :param seconds:
        :param trim:
        :return:
        """
        print('Loading Files...')
        y1 = self.load_audio(file1, trim)
        y2 = self.load_audio(file2, trim)

        print('Calculating overlap...')
        overlap = self.get_overlap(y1, y2, seconds)

        print('Calculating stfts...')
        y1ft = self.stft(y1[-overlap:])
        y2ft = self.stft(y2[:overlap])

        print('Building graph...')
        graph, nodes = self.build_graph(y1ft, y2ft, loss)
        print('Finding seam...')
        seam = self.cut(graph, nodes)

        print('Joining along seam...')
        new_slice = self.join_on_seam(y1ft, y2ft, seam)
        print('Reconstructing audio...')
        transition = self.istft(new_slice)

        print('Visualizing')
        vis = self.visualize_seam(transition, seam)

        print('DONE')
        return np.concatenate((y1[:-overlap], transition, y2[overlap:]), axis=None), transition, vis

