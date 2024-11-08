import matplotlib.pyplot as plt


class ChartFigTrajectoryCompareAll3:
    def __init__(self):
        self.tick_size = 15

    def run(self):
        ab_log_ori = [0.999309, 0.995248, 0.968069, 0.814417, 0.388457, 0.084202, 0.013133, 0.001922, 0.000278,
                      0.000040]
        ts_log_ori = [0.003, 0.016, 0.051, 0.138, 0.302, 0.492, 0.652, 0.784, 0.898, 0.998]
        ab_log_new = [0.998924, 0.995163, 0.977908, 0.832462, 0.405384, 0.091924, 0.015257, 0.002386, 0.000370,
                      0.000057]
        ts_log_new = [0.005, 0.017, 0.042, 0.130, 0.295, 0.483, 0.641, 0.771, 0.882, 0.981]
        ab_qua_ori = [0.995807, 0.970207, 0.889866, 0.723901, 0.481753, 0.236788, 0.075947, 0.013741, 0.001185,
                      0.000040]
        ts_qua_ori = [0.015, 0.049, 0.102, 0.174, 0.265, 0.374, 0.502, 0.649, 0.814, 0.998]
        ab_qua_new = [0.996924, 0.981219, 0.909756, 0.749162, 0.506046, 0.253784, 0.083935, 0.016025, 0.001543,
                      0.000068]
        ts_qua_new = [0.012, 0.038, 0.091, 0.164, 0.255, 0.365, 0.492, 0.637, 0.798, 0.972]
        ab_uni_ori = [0.895328, 0.656884, 0.394732, 0.194200, 0.078190, 0.025754, 0.006936, 0.001527, 0.000274,
                      0.000040]
        ts_uni_ori = [0.099, 0.199, 0.299, 0.399, 0.499, 0.599, 0.699, 0.799, 0.899, 0.998]
        ab_uni_new = [0.905291, 0.673222, 0.411267, 0.206440, 0.085181, 0.028907, 0.008075, 0.001858, 0.000353,
                      0.000055]
        ts_uni_new = [0.094, 0.193, 0.292, 0.391, 0.490, 0.589, 0.688, 0.786, 0.885, 0.983]
        ab_all = [0.999900, 0.999780, 0.999640, 0.999481, 0.999301, 0.999102, 0.998882, 0.998643, 0.998384, 0.998105,
                  0.997807, 0.997488, 0.997150, 0.996792, 0.996414, 0.996017, 0.995600, 0.995163, 0.994707, 0.994231,
                  0.993735, 0.993220, 0.992686, 0.992131, 0.991558, 0.990965, 0.990353, 0.989721, 0.989070, 0.988400,
                  0.987710, 0.987002, 0.986274, 0.985527, 0.984761, 0.983976, 0.983172, 0.982349, 0.981507, 0.980646,
                  0.979767, 0.978869, 0.977952, 0.977016, 0.976062, 0.975090, 0.974099, 0.973089, 0.972062, 0.971016,
                  0.969951, 0.968869, 0.967768, 0.966650, 0.965514, 0.964359, 0.963187, 0.961997, 0.960789, 0.959564,
                  0.958321, 0.957061, 0.955783, 0.954488, 0.953176, 0.951846, 0.950500, 0.949136, 0.947756, 0.946358,
                  0.944944, 0.943513, 0.942065, 0.940601, 0.939121, 0.937624, 0.936110, 0.934581, 0.933035, 0.931474,
                  0.929896, 0.928303, 0.926694, 0.925069, 0.923428, 0.921773, 0.920101, 0.918415, 0.916713, 0.914996,
                  0.913264, 0.911517, 0.909756, 0.907979, 0.906188, 0.904383, 0.902563, 0.900729, 0.898880, 0.897018,
                  0.895141, 0.893251, 0.891347, 0.889429, 0.887497, 0.885552, 0.883594, 0.881622, 0.879637, 0.877639,
                  0.875629, 0.873605, 0.871569, 0.869520, 0.867458, 0.865384, 0.863298, 0.861200, 0.859089, 0.856967,
                  0.854832, 0.852687, 0.850529, 0.848360, 0.846180, 0.843988, 0.841785, 0.839572, 0.837347, 0.835112,
                  0.832865, 0.830609, 0.828342, 0.826064, 0.823777, 0.821479, 0.819171, 0.816854, 0.814527, 0.812190,
                  0.809844, 0.807488, 0.805123, 0.802750, 0.800367, 0.797975, 0.795574, 0.793165, 0.790747, 0.788321,
                  0.785887, 0.783444, 0.780994, 0.778536, 0.776069, 0.773596, 0.771114, 0.768626, 0.766130, 0.763626,
                  0.761116, 0.758599, 0.756075, 0.753545, 0.751008, 0.748464, 0.745914, 0.743358, 0.740797, 0.738229,
                  0.735655, 0.733075, 0.730490, 0.727900, 0.725304, 0.722703, 0.720097, 0.717486, 0.714871, 0.712250,
                  0.709625, 0.706995, 0.704362, 0.701724, 0.699081, 0.696435, 0.693785, 0.691131, 0.688474, 0.685813,
                  0.683149, 0.680481, 0.677811, 0.675137, 0.672461, 0.669782, 0.667099, 0.664415, 0.661728, 0.659039,
                  0.656347, 0.653654, 0.650958, 0.648261, 0.645561, 0.642861, 0.640158, 0.637455, 0.634750, 0.632044,
                  0.629336, 0.626628, 0.623919, 0.621210, 0.618500, 0.615789, 0.613078, 0.610366, 0.607655, 0.604943,
                  0.602232, 0.599520, 0.596809, 0.594098, 0.591388, 0.588678, 0.585969, 0.583261, 0.580554, 0.577847,
                  0.575142, 0.572438, 0.569735, 0.567034, 0.564334, 0.561636, 0.558940, 0.556245, 0.553552, 0.550861,
                  0.548173, 0.545486, 0.542802, 0.540120, 0.537441, 0.534764, 0.532090, 0.529419, 0.526751, 0.524085,
                  0.521423, 0.518764, 0.516108, 0.513455, 0.510806, 0.508160, 0.505518, 0.502880, 0.500245, 0.497614,
                  0.494987, 0.492364, 0.489745, 0.487130, 0.484520, 0.481914, 0.479312, 0.476715, 0.474122, 0.471534,
                  0.468951, 0.466372, 0.463799, 0.461230, 0.458667, 0.456108, 0.453555, 0.451007, 0.448464, 0.445927,
                  0.443395, 0.440869, 0.438348, 0.435833, 0.433324, 0.430821, 0.428324, 0.425832, 0.423346, 0.420867,
                  0.418394, 0.415926, 0.413466, 0.411011, 0.408563, 0.406121, 0.403686, 0.401257, 0.398835, 0.396420,
                  0.394011, 0.391609, 0.389214, 0.386826, 0.384445, 0.382071, 0.379704, 0.377344, 0.374991, 0.372645,
                  0.370307, 0.367976, 0.365652, 0.363335, 0.361027, 0.358725, 0.356431, 0.354145, 0.351866, 0.349595,
                  0.347331, 0.345076, 0.342828, 0.340588, 0.338356, 0.336131, 0.333915, 0.331706, 0.329506, 0.327314,
                  0.325129, 0.322953, 0.320785, 0.318625, 0.316473, 0.314329, 0.312194, 0.310067, 0.307949, 0.305838,
                  0.303736, 0.301643, 0.299558, 0.297481, 0.295413, 0.293353, 0.291302, 0.289259, 0.287225, 0.285200,
                  0.283183, 0.281174, 0.279175, 0.277184, 0.275201, 0.273228, 0.271263, 0.269307, 0.267359, 0.265420,
                  0.263490, 0.261569, 0.259657, 0.257754, 0.255859, 0.253973, 0.252096, 0.250228, 0.248368, 0.246518,
                  0.244676, 0.242844, 0.241020, 0.239205, 0.237399, 0.235602, 0.233814, 0.232034, 0.230264, 0.228503,
                  0.226750, 0.225007, 0.223272, 0.221546, 0.219829, 0.218121, 0.216422, 0.214732, 0.213051, 0.211379,
                  0.209716, 0.208061, 0.206416, 0.204779, 0.203152, 0.201533, 0.199923, 0.198322, 0.196730, 0.195146,
                  0.193572, 0.192006, 0.190450, 0.188902, 0.187363, 0.185832, 0.184311, 0.182798, 0.181294, 0.179799,
                  0.178313, 0.176835, 0.175366, 0.173906, 0.172454, 0.171011, 0.169577, 0.168152, 0.166735, 0.165326,
                  0.163927, 0.162535, 0.161153, 0.159779, 0.158413, 0.157056, 0.155708, 0.154368, 0.153036, 0.151713,
                  0.150399, 0.149092, 0.147794, 0.146505, 0.145224, 0.143951, 0.142686, 0.141430, 0.140182, 0.138942,
                  0.137710, 0.136487, 0.135271, 0.134064, 0.132865, 0.131674, 0.130491, 0.129316, 0.128149, 0.126990,
                  0.125839, 0.124696, 0.123561, 0.122433, 0.121314, 0.120202, 0.119098, 0.118002, 0.116914, 0.115833,
                  0.114760, 0.113695, 0.112637, 0.111587, 0.110544, 0.109509, 0.108482, 0.107462, 0.106449, 0.105444,
                  0.104447, 0.103456, 0.102473, 0.101497, 0.100529, 0.099568, 0.098614, 0.097667, 0.096727, 0.095794,
                  0.094869, 0.093950, 0.093039, 0.092134, 0.091237, 0.090346, 0.089463, 0.088586, 0.087716, 0.086853,
                  0.085996, 0.085146, 0.084303, 0.083467, 0.082637, 0.081814, 0.080998, 0.080188, 0.079384, 0.078587,
                  0.077797, 0.077013, 0.076235, 0.075463, 0.074698, 0.073939, 0.073186, 0.072440, 0.071700, 0.070966,
                  0.070238, 0.069516, 0.068800, 0.068090, 0.067386, 0.066688, 0.065996, 0.065309, 0.064629, 0.063954,
                  0.063285, 0.062622, 0.061965, 0.061313, 0.060667, 0.060026, 0.059392, 0.058762, 0.058138, 0.057520,
                  0.056907, 0.056299, 0.055697, 0.055100, 0.054508, 0.053922, 0.053341, 0.052765, 0.052194, 0.051629,
                  0.051068, 0.050513, 0.049962, 0.049417, 0.048876, 0.048341, 0.047810, 0.047284, 0.046764, 0.046247,
                  0.045736, 0.045230, 0.044728, 0.044231, 0.043738, 0.043250, 0.042767, 0.042288, 0.041814, 0.041344,
                  0.040879, 0.040418, 0.039961, 0.039509, 0.039061, 0.038618, 0.038178, 0.037743, 0.037313, 0.036886,
                  0.036463, 0.036045, 0.035631, 0.035220, 0.034814, 0.034412, 0.034014, 0.033619, 0.033229, 0.032842,
                  0.032460, 0.032081, 0.031706, 0.031334, 0.030966, 0.030603, 0.030242, 0.029886, 0.029533, 0.029183,
                  0.028837, 0.028495, 0.028156, 0.027821, 0.027489, 0.027160, 0.026835, 0.026513, 0.026195, 0.025879,
                  0.025567, 0.025259, 0.024953, 0.024651, 0.024352, 0.024056, 0.023763, 0.023474, 0.023187, 0.022903,
                  0.022623, 0.022345, 0.022071, 0.021799, 0.021530, 0.021264, 0.021001, 0.020741, 0.020484, 0.020229,
                  0.019977, 0.019728, 0.019482, 0.019238, 0.018997, 0.018758, 0.018523, 0.018289, 0.018059, 0.017831,
                  0.017605, 0.017382, 0.017161, 0.016943, 0.016728, 0.016514, 0.016304, 0.016095, 0.015889, 0.015685,
                  0.015484, 0.015284, 0.015087, 0.014893, 0.014700, 0.014510, 0.014321, 0.014135, 0.013952, 0.013770,
                  0.013590, 0.013413, 0.013237, 0.013064, 0.012892, 0.012723, 0.012555, 0.012389, 0.012226, 0.012064,
                  0.011904, 0.011746, 0.011590, 0.011436, 0.011284, 0.011133, 0.010984, 0.010837, 0.010692, 0.010548,
                  0.010407, 0.010266, 0.010128, 0.009991, 0.009856, 0.009722, 0.009591, 0.009460, 0.009332, 0.009204,
                  0.009079, 0.008955, 0.008832, 0.008711, 0.008592, 0.008474, 0.008357, 0.008242, 0.008128, 0.008016,
                  0.007905, 0.007795, 0.007687, 0.007580, 0.007474, 0.007370, 0.007267, 0.007166, 0.007065, 0.006966,
                  0.006868, 0.006772, 0.006676, 0.006582, 0.006489, 0.006397, 0.006307, 0.006217, 0.006129, 0.006042,
                  0.005956, 0.005871, 0.005787, 0.005704, 0.005623, 0.005542, 0.005462, 0.005384, 0.005306, 0.005230,
                  0.005154, 0.005080, 0.005006, 0.004933, 0.004862, 0.004791, 0.004721, 0.004652, 0.004585, 0.004518,
                  0.004451, 0.004386, 0.004322, 0.004258, 0.004195, 0.004134, 0.004073, 0.004012, 0.003953, 0.003894,
                  0.003837, 0.003780, 0.003723, 0.003668, 0.003613, 0.003559, 0.003506, 0.003453, 0.003402, 0.003351,
                  0.003300, 0.003250, 0.003201, 0.003153, 0.003105, 0.003058, 0.003012, 0.002966, 0.002921, 0.002877,
                  0.002833, 0.002790, 0.002747, 0.002705, 0.002664, 0.002623, 0.002582, 0.002543, 0.002504, 0.002465,
                  0.002427, 0.002389, 0.002352, 0.002316, 0.002280, 0.002245, 0.002210, 0.002175, 0.002141, 0.002108,
                  0.002075, 0.002042, 0.002010, 0.001979, 0.001948, 0.001917, 0.001887, 0.001857, 0.001828, 0.001799,
                  0.001770, 0.001742, 0.001715, 0.001687, 0.001661, 0.001634, 0.001608, 0.001582, 0.001557, 0.001532,
                  0.001508, 0.001483, 0.001459, 0.001436, 0.001413, 0.001390, 0.001368, 0.001345, 0.001324, 0.001302,
                  0.001281, 0.001260, 0.001240, 0.001220, 0.001200, 0.001180, 0.001161, 0.001142, 0.001123, 0.001105,
                  0.001086, 0.001069, 0.001051, 0.001034, 0.001017, 0.001000, 0.000983, 0.000967, 0.000951, 0.000935,
                  0.000919, 0.000904, 0.000889, 0.000874, 0.000860, 0.000845, 0.000831, 0.000817, 0.000803, 0.000790,
                  0.000777, 0.000764, 0.000751, 0.000738, 0.000726, 0.000713, 0.000701, 0.000689, 0.000678, 0.000666,
                  0.000655, 0.000643, 0.000633, 0.000622, 0.000611, 0.000601, 0.000590, 0.000580, 0.000570, 0.000560,
                  0.000551, 0.000541, 0.000532, 0.000523, 0.000514, 0.000505, 0.000496, 0.000487, 0.000479, 0.000471,
                  0.000462, 0.000454, 0.000446, 0.000439, 0.000431, 0.000423, 0.000416, 0.000409, 0.000401, 0.000394,
                  0.000387, 0.000381, 0.000374, 0.000367, 0.000361, 0.000354, 0.000348, 0.000342, 0.000336, 0.000330,
                  0.000324, 0.000318, 0.000312, 0.000307, 0.000301, 0.000296, 0.000291, 0.000285, 0.000280, 0.000275,
                  0.000270, 0.000265, 0.000261, 0.000256, 0.000251, 0.000247, 0.000242, 0.000238, 0.000233, 0.000229,
                  0.000225, 0.000221, 0.000217, 0.000213, 0.000209, 0.000205, 0.000201, 0.000198, 0.000194, 0.000191,
                  0.000187, 0.000184, 0.000180, 0.000177, 0.000174, 0.000170, 0.000167, 0.000164, 0.000161, 0.000158,
                  0.000155, 0.000152, 0.000149, 0.000147, 0.000144, 0.000141, 0.000139, 0.000136, 0.000133, 0.000131,
                  0.000128, 0.000126, 0.000124, 0.000121, 0.000119, 0.000117, 0.000114, 0.000112, 0.000110, 0.000108,
                  0.000106, 0.000104, 0.000102, 0.000100, 0.000098, 0.000096, 0.000094, 0.000093, 0.000091, 0.000089,
                  0.000087, 0.000086, 0.000084, 0.000082, 0.000081, 0.000079, 0.000078, 0.000076, 0.000075, 0.000073,
                  0.000072, 0.000071, 0.000069, 0.000068, 0.000066, 0.000065, 0.000064, 0.000063, 0.000061, 0.000060,
                  0.000059, 0.000058, 0.000057, 0.000056, 0.000055, 0.000053, 0.000052, 0.000051, 0.000050, 0.000049,
                  0.000048, 0.000047, 0.000046, 0.000046, 0.000045, 0.000044, 0.000043, 0.000042, 0.000041, 0.000040]
        ts_all = list(range(len(ab_all)))
        ts_all = [(float(i)) / 1000.0 for i in ts_all]
        abo_arr = [ab_log_ori, ab_qua_ori, ab_uni_ori]
        tso_arr = [ts_log_ori, ts_qua_ori, ts_uni_ori]
        abn_arr = [ab_log_new, ab_qua_new, ab_uni_new]
        tsn_arr = [ts_log_new, ts_qua_new, ts_uni_new]
        ttl_arr = ['logSNR', 'quadratic', 'uniform']

        tick_size = self.tick_size
        legend_size = 15
        xy_label_size = 20
        title_size = 20
        fig = plt.figure(figsize=(12, 7))
        ax_arr = fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)
        for ax, ab_ori, ts_ori, ab_new, ts_new, ttl in zip(ax_arr, abo_arr, tso_arr, abn_arr, tsn_arr, ttl_arr):
            ax.tick_params(axis='y', labelsize=tick_size)
            ax.xaxis.set_ticklabels([])
            ax.set_xlim((0, 1.02))
            ax.set_ylim((-0.05, 1.05))
            ax.plot(ts_ori, ab_ori, 'o', ms=6, color='b')
            ax.plot(ts_new, ab_new, '*', ms=6, color='red')
            ax.plot(ts_all, ab_all, '--', color='green', linewidth=1)
            # why duplicate: the ax legend only follow the plot order, but we want marker cover line.
            ax.plot(ts_ori, ab_ori, 'o', ms=6, color='b')
            ax.plot(ts_new, ab_new, '*', ms=6, color='red')
            ax.legend(['Initial trajectory', 'Optimized trajectory'], fontsize=legend_size, loc='upper right')
            ax.text(0.45, 0.9, ttl, size=title_size)
        # for
        fig.supylabel(r"$\bar{\alpha}$", fontsize=xy_label_size, rotation=0)  # make it horizontal
        fig.supxlabel(r"timestep $t$", fontsize=xy_label_size)
        fig.tight_layout()
        plt.text(0.00, -0.25, r"$0$", size=20)
        plt.text(0.99, -0.25, r"$T$", size=20)
        f_path = './configs/chart_aaai2025/abc_o1_step10_all3/fig_abc_o1_s10_all3.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"Saved: {f_path}")
        plt.close()

        fig = plt.figure(figsize=(6, 4))  # ------------------------- sup
        ax = fig.add_subplot(111)
        ab_log_ori = ab_log_ori[:3]
        ts_log_ori = ts_log_ori[:3]
        ab_log_new = ab_log_new[:3]
        ts_log_new = ts_log_new[:3]
        ab_last = min(ab_log_ori[-1], ab_log_new[-1])
        idx = 0
        while ab_all[idx] >= ab_last:
            idx += 1
        ab_all = ab_all[:idx + 1]
        ts_all = ts_all[:idx + 1]
        ab_log_ori = [f + 0.0004 for f in ab_log_ori]  # small adjustment based on observation
        ax.tick_params('y', labelsize=tick_size)
        ax.xaxis.set_ticklabels([])
        ax.plot(ts_log_ori, ab_log_ori, 'o', ms=6, color='b')
        ax.plot(ts_log_new, ab_log_new, '*', ms=6, color='red')
        ax.plot(ts_all, ab_all, '--', color='green', linewidth=1)
        # why duplicate: the ax legend only follow the plot order, but we want marker cover line.
        ax.plot(ts_log_ori, ab_log_ori, 'o', ms=6, color='b')
        ax.plot(ts_log_new, ab_log_new, '*', ms=6, color='red')
        fig.supylabel(r"$\bar{\alpha}$", fontsize=xy_label_size, rotation=0)  # make it horizontal
        fig.supxlabel(r"timestep $t$", fontsize=xy_label_size)
        fig.tight_layout()
        plt.text(0.028, -0.1, r"$0$", size=20, transform=ax.transAxes)
        f_path = './configs/chart_aaai2025/abc_o1_step10_all3/fig_abc_o1_s10_all3_sup.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"Saved: {f_path}")
        plt.close()

# class