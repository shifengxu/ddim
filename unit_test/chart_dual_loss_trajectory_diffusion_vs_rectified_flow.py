import matplotlib.pyplot as plt


class ChartDualLossTrajectoryDiffusionVsRectifiedFlow:
    def __init__(self):
        self.fig_size = (16, 8)

    def run(self):
        x1k_df = [0.257086, 0.257072, 0.257058, 0.257044, 0.257030, 0.257017, 0.257004, 0.256990, 0.256977, 0.256963,
                  0.256949, 0.256935, 0.256921, 0.256907, 0.256892, 0.256877, 0.256862, 0.256848, 0.256833, 0.256818,
                  0.256802, 0.256787, 0.256772, 0.256757, 0.256742, 0.256727, 0.256711, 0.256696, 0.256680, 0.256663,
                  0.256646, 0.256629, 0.256612, 0.256596, 0.256579, 0.256562, 0.256545, 0.256528, 0.256511, 0.256492,
                  0.256474, 0.256455, 0.256436, 0.256417, 0.256398, 0.256378, 0.256359, 0.256340, 0.256322, 0.256302,
                  0.256283, 0.256263, 0.256244, 0.256224, 0.256204, 0.256184, 0.256163, 0.256141, 0.256120, 0.256098,
                  0.256075, 0.256053, 0.256031, 0.256008, 0.255985, 0.255961, 0.255937, 0.255913, 0.255889, 0.255865,
                  0.255840, 0.255815, 0.255789, 0.255765, 0.255739, 0.255713, 0.255686, 0.255659, 0.255631, 0.255603,
                  0.255575, 0.255546, 0.255517, 0.255489, 0.255459, 0.255430, 0.255400, 0.255369, 0.255338, 0.255307,
                  0.255275, 0.255243, 0.255211, 0.255177, 0.255144, 0.255110, 0.255076, 0.255041, 0.255005, 0.254969,
                  0.254933, 0.254897, 0.254860, 0.254822, 0.254784, 0.254746, 0.254707, 0.254668, 0.254628, 0.254587,
                  0.254545, 0.254504, 0.254463, 0.254421, 0.254378, 0.254334, 0.254290, 0.254245, 0.254199, 0.254154,
                  0.254107, 0.254060, 0.254012, 0.253965, 0.253916, 0.253868, 0.253818, 0.253768, 0.253718, 0.253668,
                  0.253616, 0.253564, 0.253512, 0.253458, 0.253405, 0.253351, 0.253296, 0.253241, 0.253186, 0.253130,
                  0.253075, 0.253018, 0.252961, 0.252903, 0.252845, 0.252785, 0.252726, 0.252665, 0.252605, 0.252544,
                  0.252483, 0.252421, 0.252358, 0.252295, 0.252231, 0.252167, 0.252102, 0.252036, 0.251969, 0.251902,
                  0.251835, 0.251767, 0.251698, 0.251629, 0.251560, 0.251490, 0.251419, 0.251348, 0.251276, 0.251203,
                  0.251130, 0.251056, 0.250981, 0.250906, 0.250831, 0.250755, 0.250679, 0.250603, 0.250526, 0.250448,
                  0.250371, 0.250293, 0.250214, 0.250135, 0.250055, 0.249975, 0.249894, 0.249814, 0.249733, 0.249652,
                  0.249571, 0.249489, 0.249407, 0.249325, 0.249243, 0.249161, 0.249079, 0.248996, 0.248914, 0.248832,
                  0.248749, 0.248667, 0.248586, 0.248504, 0.248423, 0.248342, 0.248260, 0.248179, 0.248098, 0.248018,
                  0.247938, 0.247859, 0.247780, 0.247701, 0.247623, 0.247546, 0.247469, 0.247393, 0.247318, 0.247243,
                  0.247169, 0.247096, 0.247024, 0.246952, 0.246882, 0.246813, 0.246745, 0.246678, 0.246612, 0.246547,
                  0.246484, 0.246422, 0.246362, 0.246304, 0.246248, 0.246193, 0.246139, 0.246086, 0.246036, 0.245987,
                  0.245941, 0.245897, 0.245854, 0.245813, 0.245774, 0.245737, 0.245702, 0.245668, 0.245637, 0.245607,
                  0.245581, 0.245557, 0.245535, 0.245515, 0.245498, 0.245483, 0.245471, 0.245461, 0.245453, 0.245448,
                  0.245447, 0.245448, 0.245452, 0.245459, 0.245468, 0.245480, 0.245496, 0.245514, 0.245534, 0.245558,
                  0.245585, 0.245615, 0.245647, 0.245682, 0.245721, 0.245762, 0.245807, 0.245854, 0.245904, 0.245958,
                  0.246014, 0.246072, 0.246134, 0.246199, 0.246267, 0.246339, 0.246413, 0.246490, 0.246570, 0.246652,
                  0.246737, 0.246826, 0.246918, 0.247013, 0.247110, 0.247210, 0.247313, 0.247420, 0.247530, 0.247643,
                  0.247760, 0.247879, 0.248002, 0.248127, 0.248257, 0.248390, 0.248527, 0.248666, 0.248809, 0.248955,
                  0.249104, 0.249257, 0.249413, 0.249573, 0.249736, 0.249903, 0.250073, 0.250247, 0.250425, 0.250607,
                  0.250793, 0.250982, 0.251174, 0.251370, 0.251571, 0.251776, 0.251985, 0.252199, 0.252416, 0.252638,
                  0.252864, 0.253094, 0.253330, 0.253570, 0.253815, 0.254064, 0.254316, 0.254574, 0.254835, 0.255102,
                  0.255373, 0.255650, 0.255932, 0.256219, 0.256510, 0.256806, 0.257105, 0.257411, 0.257721, 0.258036,
                  0.258355, 0.258678, 0.259006, 0.259340, 0.259678, 0.260020, 0.260368, 0.260721, 0.261077, 0.261440,
                  0.261806, 0.262177, 0.262553, 0.262934, 0.263318, 0.263708, 0.264103, 0.264501, 0.264903, 0.265310,
                  0.265721, 0.266136, 0.266555, 0.266979, 0.267408, 0.267840, 0.268277, 0.268718, 0.269164, 0.269614,
                  0.270068, 0.270527, 0.270990, 0.271457, 0.271929, 0.272404, 0.272882, 0.273365, 0.273851, 0.274341,
                  0.274835, 0.275332, 0.275833, 0.276339, 0.276848, 0.277361, 0.277877, 0.278396, 0.278917, 0.279443,
                  0.279972, 0.280505, 0.281041, 0.281580, 0.282122, 0.282666, 0.283215, 0.283766, 0.284321, 0.284879,
                  0.285441, 0.286005, 0.286573, 0.287144, 0.287718, 0.288296, 0.288877, 0.289460, 0.290046, 0.290634,
                  0.291227, 0.291822, 0.292420, 0.293019, 0.293622, 0.294228, 0.294837, 0.295449, 0.296063, 0.296679,
                  0.297298, 0.297919, 0.298544, 0.299172, 0.299802, 0.300433, 0.301067, 0.301705, 0.302345, 0.302989,
                  0.303634, 0.304282, 0.304932, 0.305584, 0.306239, 0.306895, 0.307554, 0.308216, 0.308879, 0.309546,
                  0.310214, 0.310885, 0.311558, 0.312234, 0.312912, 0.313592, 0.314275, 0.314959, 0.315645, 0.316334,
                  0.317025, 0.317718, 0.318413, 0.319110, 0.319809, 0.320511, 0.321215, 0.321920, 0.322628, 0.323336,
                  0.324047, 0.324760, 0.325476, 0.326193, 0.326911, 0.327632, 0.328355, 0.329080, 0.329806, 0.330534,
                  0.331264, 0.331995, 0.332728, 0.333462, 0.334199, 0.334938, 0.335678, 0.336420, 0.337163, 0.337908,
                  0.338654, 0.339402, 0.340152, 0.340902, 0.341655, 0.342409, 0.343164, 0.343921, 0.344679, 0.345439,
                  0.346199, 0.346961, 0.347725, 0.348491, 0.349258, 0.350026, 0.350795, 0.351567, 0.352339, 0.353112,
                  0.353887, 0.354662, 0.355439, 0.356217, 0.356995, 0.357775, 0.358555, 0.359336, 0.360119, 0.360902,
                  0.361688, 0.362474, 0.363260, 0.364048, 0.364838, 0.365628, 0.366419, 0.367211, 0.368004, 0.368798,
                  0.369593, 0.370388, 0.371184, 0.371981, 0.372779, 0.373578, 0.374377, 0.375177, 0.375978, 0.376779,
                  0.377582, 0.378385, 0.379189, 0.379993, 0.380798, 0.381604, 0.382410, 0.383217, 0.384025, 0.384833,
                  0.385641, 0.386450, 0.387259, 0.388069, 0.388879, 0.389691, 0.390503, 0.391315, 0.392127, 0.392940,
                  0.393754, 0.394568, 0.395382, 0.396196, 0.397011, 0.397826, 0.398641, 0.399457, 0.400272, 0.401089,
                  0.401905, 0.402721, 0.403537, 0.404353, 0.405168, 0.405985, 0.406802, 0.407618, 0.408435, 0.409252,
                  0.410068, 0.410885, 0.411701, 0.412518, 0.413335, 0.414152, 0.414968, 0.415784, 0.416600, 0.417417,
                  0.418234, 0.419050, 0.419865, 0.420681, 0.421497, 0.422313, 0.423129, 0.423945, 0.424759, 0.425574,
                  0.426389, 0.427203, 0.428018, 0.428832, 0.429644, 0.430457, 0.431269, 0.432082, 0.432894, 0.433706,
                  0.434517, 0.435329, 0.436139, 0.436949, 0.437758, 0.438566, 0.439375, 0.440184, 0.440991, 0.441798,
                  0.442605, 0.443412, 0.444218, 0.445023, 0.445828, 0.446634, 0.447438, 0.448241, 0.449044, 0.449846,
                  0.450647, 0.451447, 0.452247, 0.453046, 0.453844, 0.454640, 0.455437, 0.456233, 0.457028, 0.457823,
                  0.458617, 0.459411, 0.460203, 0.460995, 0.461785, 0.462576, 0.463364, 0.464152, 0.464940, 0.465727,
                  0.466512, 0.467296, 0.468080, 0.468862, 0.469643, 0.470424, 0.471204, 0.471982, 0.472760, 0.473538,
                  0.474313, 0.475088, 0.475862, 0.476634, 0.477405, 0.478175, 0.478945, 0.479712, 0.480479, 0.481246,
                  0.482010, 0.482774, 0.483537, 0.484298, 0.485058, 0.485818, 0.486576, 0.487332, 0.488088, 0.488842,
                  0.489594, 0.490347, 0.491096, 0.491844, 0.492593, 0.493338, 0.494083, 0.494828, 0.495569, 0.496310,
                  0.497050, 0.497788, 0.498525, 0.499261, 0.499994, 0.500728, 0.501458, 0.502187, 0.502915, 0.503640,
                  0.504365, 0.505088, 0.505809, 0.506530, 0.507249, 0.507967, 0.508684, 0.509397, 0.510110, 0.510821,
                  0.511531, 0.512240, 0.512945, 0.513650, 0.514354, 0.515055, 0.515756, 0.516453, 0.517151, 0.517846,
                  0.518540, 0.519233, 0.519922, 0.520612, 0.521298, 0.521983, 0.522667, 0.523349, 0.524031, 0.524708,
                  0.525386, 0.526060, 0.526733, 0.527404, 0.528073, 0.528742, 0.529406, 0.530072, 0.530733, 0.531394,
                  0.532053, 0.532710, 0.533366, 0.534020, 0.534673, 0.535322, 0.535971, 0.536616, 0.537263, 0.537904,
                  0.538546, 0.539183, 0.539821, 0.540454, 0.541088, 0.541718, 0.542347, 0.542973, 0.543598, 0.544221,
                  0.544841, 0.545461, 0.546078, 0.546694, 0.547307, 0.547919, 0.548528, 0.549136, 0.549741, 0.550345,
                  0.550946, 0.551546, 0.552143, 0.552739, 0.553332, 0.553924, 0.554513, 0.555100, 0.555686, 0.556269,
                  0.556852, 0.557430, 0.558009, 0.558583, 0.559160, 0.559729, 0.560300, 0.560866, 0.561432, 0.561995,
                  0.562557, 0.563117, 0.563675, 0.564232, 0.564784, 0.565338, 0.565885, 0.566435, 0.566979, 0.567523,
                  0.568064, 0.568603, 0.569142, 0.569676, 0.570211, 0.570740, 0.571270, 0.571797, 0.572322, 0.572847,
                  0.573366, 0.573888, 0.574404, 0.574920, 0.575434, 0.575944, 0.576457, 0.576962, 0.577468, 0.577972,
                  0.578473, 0.578976, 0.579470, 0.579967, 0.580461, 0.580951, 0.581444, 0.581929, 0.582416, 0.582901,
                  0.583381, 0.583864, 0.584341, 0.584817, 0.585295, 0.585764, 0.586235, 0.586705, 0.587169, 0.587636,
                  0.588099, 0.588558, 0.589020, 0.589475, 0.589930, 0.590387, 0.590834, 0.591284, 0.591735, 0.592176,
                  0.592621, 0.593066, 0.593502, 0.593941, 0.594380, 0.594811, 0.595244, 0.595677, 0.596102, 0.596529,
                  0.596957, 0.597376, 0.597797, 0.598221, 0.598633, 0.599048, 0.599466, 0.599874, 0.600283, 0.600694,
                  0.601099, 0.601501, 0.601906, 0.602308, 0.602703, 0.603101, 0.603501, 0.603890, 0.604280, 0.604672,
                  0.605060, 0.605442, 0.605827, 0.606214, 0.606589, 0.606966, 0.607346, 0.607722, 0.608091, 0.608462,
                  0.608836, 0.609201, 0.609564, 0.609929, 0.610296, 0.610652, 0.611007, 0.611365, 0.611726, 0.612072,
                  0.612420, 0.612771, 0.613124, 0.613462, 0.613803, 0.614146, 0.614491, 0.614824, 0.615156, 0.615490,
                  0.615827, 0.616156, 0.616479, 0.616805, 0.617132, 0.617460, 0.617774, 0.618090, 0.618408, 0.618728,
                  0.619041, 0.619346, 0.619653, 0.619962, 0.620273, 0.620573, 0.620869, 0.621166, 0.621465, 0.621766,
                  0.622057, 0.622342, 0.622629, 0.622916, 0.623206, 0.623492, 0.623765, 0.624039, 0.624315, 0.624592,
                  0.624871, 0.625137, 0.625397, 0.625659, 0.625922, 0.626187, 0.626453, 0.626703, 0.626950, 0.627196,
                  0.627445, 0.627694, 0.627943, 0.628186, 0.628415, 0.628644, 0.628873, 0.629103, 0.629332, 0.629562,
                  0.629784, 0.629991, 0.630198, 0.630405, 0.630611, 0.630817, 0.631021, 0.631226, 0.631415, 0.631593,
                  0.631771, 0.631949, 0.632123, 0.632296, 0.632467, 0.632636, 0.632802, 0.632948, 0.633088, 0.633228,
                  0.633363, 0.633498, 0.633629, 0.633758, 0.633885, 0.634010, 0.634134, 0.634245, 0.634347, 0.634450,
                  0.634550, 0.634650, 0.634750, 0.634849, 0.634949, 0.635048, 0.635148, 0.635248, 0.635348, 0.635449,
                  0.635549]  # diffusion
        y1k_df = [0.952998, 0.952969, 0.952942, 0.952916, 0.952891, 0.952865, 0.952839, 0.952813, 0.952787, 0.952760,
                  0.952733, 0.952707, 0.952680, 0.952653, 0.952626, 0.952598, 0.952569, 0.952541, 0.952513, 0.952484,
                  0.952455, 0.952425, 0.952394, 0.952363, 0.952332, 0.952301, 0.952271, 0.952241, 0.952210, 0.952177,
                  0.952145, 0.952114, 0.952081, 0.952048, 0.952016, 0.951982, 0.951948, 0.951913, 0.951876, 0.951840,
                  0.951803, 0.951767, 0.951730, 0.951692, 0.951654, 0.951616, 0.951577, 0.951538, 0.951499, 0.951460,
                  0.951420, 0.951380, 0.951339, 0.951297, 0.951255, 0.951213, 0.951168, 0.951123, 0.951078, 0.951032,
                  0.950987, 0.950941, 0.950893, 0.950844, 0.950795, 0.950745, 0.950695, 0.950644, 0.950594, 0.950542,
                  0.950490, 0.950437, 0.950383, 0.950328, 0.950272, 0.950216, 0.950158, 0.950099, 0.950039, 0.949979,
                  0.949917, 0.949854, 0.949792, 0.949728, 0.949664, 0.949598, 0.949532, 0.949464, 0.949395, 0.949325,
                  0.949254, 0.949182, 0.949109, 0.949035, 0.948959, 0.948883, 0.948806, 0.948727, 0.948647, 0.948565,
                  0.948483, 0.948400, 0.948316, 0.948232, 0.948146, 0.948058, 0.947969, 0.947879, 0.947788, 0.947694,
                  0.947601, 0.947506, 0.947410, 0.947311, 0.947212, 0.947112, 0.947011, 0.946909, 0.946805, 0.946700,
                  0.946593, 0.946485, 0.946375, 0.946263, 0.946150, 0.946036, 0.945920, 0.945802, 0.945683, 0.945562,
                  0.945440, 0.945316, 0.945190, 0.945063, 0.944935, 0.944806, 0.944675, 0.944542, 0.944408, 0.944272,
                  0.944135, 0.943995, 0.943853, 0.943709, 0.943564, 0.943417, 0.943270, 0.943120, 0.942968, 0.942813,
                  0.942658, 0.942500, 0.942341, 0.942179, 0.942015, 0.941849, 0.941681, 0.941510, 0.941336, 0.941159,
                  0.940981, 0.940800, 0.940618, 0.940433, 0.940247, 0.940058, 0.939866, 0.939671, 0.939474, 0.939275,
                  0.939072, 0.938867, 0.938660, 0.938450, 0.938238, 0.938024, 0.937806, 0.937586, 0.937363, 0.937137,
                  0.936909, 0.936678, 0.936444, 0.936207, 0.935966, 0.935723, 0.935477, 0.935228, 0.934976, 0.934720,
                  0.934461, 0.934199, 0.933936, 0.933668, 0.933397, 0.933123, 0.932845, 0.932565, 0.932281, 0.931994,
                  0.931702, 0.931408, 0.931110, 0.930809, 0.930505, 0.930198, 0.929886, 0.929570, 0.929251, 0.928927,
                  0.928600, 0.928269, 0.927934, 0.927596, 0.927254, 0.926907, 0.926556, 0.926202, 0.925844, 0.925483,
                  0.925117, 0.924746, 0.924371, 0.923992, 0.923609, 0.923223, 0.922832, 0.922436, 0.922036, 0.921632,
                  0.921224, 0.920812, 0.920393, 0.919969, 0.919541, 0.919110, 0.918676, 0.918237, 0.917792, 0.917342,
                  0.916888, 0.916428, 0.915965, 0.915496, 0.915022, 0.914544, 0.914062, 0.913575, 0.913083, 0.912587,
                  0.912085, 0.911577, 0.911066, 0.910549, 0.910028, 0.909502, 0.908971, 0.908435, 0.907894, 0.907347,
                  0.906795, 0.906238, 0.905675, 0.905108, 0.904536, 0.903958, 0.903375, 0.902787, 0.902195, 0.901596,
                  0.900993, 0.900385, 0.899773, 0.899156, 0.898534, 0.897906, 0.897274, 0.896637, 0.895995, 0.895348,
                  0.894696, 0.894040, 0.893378, 0.892712, 0.892040, 0.891364, 0.890682, 0.889996, 0.889306, 0.888612,
                  0.887913, 0.887209, 0.886500, 0.885786, 0.885067, 0.884344, 0.883617, 0.882884, 0.882147, 0.881405,
                  0.880659, 0.879907, 0.879151, 0.878389, 0.877622, 0.876850, 0.876073, 0.875292, 0.874506, 0.873714,
                  0.872918, 0.872117, 0.871310, 0.870499, 0.869683, 0.868862, 0.868036, 0.867205, 0.866367, 0.865525,
                  0.864677, 0.863825, 0.862968, 0.862106, 0.861237, 0.860364, 0.859484, 0.858599, 0.857710, 0.856815,
                  0.855915, 0.855008, 0.854096, 0.853178, 0.852255, 0.851326, 0.850392, 0.849452, 0.848508, 0.847558,
                  0.846602, 0.845641, 0.844672, 0.843698, 0.842719, 0.841734, 0.840744, 0.839748, 0.838747, 0.837740,
                  0.836729, 0.835713, 0.834690, 0.833662, 0.832628, 0.831588, 0.830543, 0.829492, 0.828436, 0.827373,
                  0.826306, 0.825233, 0.824154, 0.823070, 0.821980, 0.820885, 0.819784, 0.818677, 0.817566, 0.816448,
                  0.815325, 0.814198, 0.813065, 0.811927, 0.810784, 0.809635, 0.808482, 0.807323, 0.806158, 0.804989,
                  0.803814, 0.802633, 0.801448, 0.800258, 0.799062, 0.797861, 0.796656, 0.795446, 0.794231, 0.793011,
                  0.791786, 0.790557, 0.789323, 0.788083, 0.786839, 0.785590, 0.784337, 0.783080, 0.781819, 0.780554,
                  0.779283, 0.778008, 0.776729, 0.775446, 0.774159, 0.772869, 0.771573, 0.770273, 0.768969, 0.767661,
                  0.766349, 0.765033, 0.763713, 0.762389, 0.761061, 0.759729, 0.758394, 0.757055, 0.755713, 0.754368,
                  0.753017, 0.751664, 0.750307, 0.748947, 0.747584, 0.746217, 0.744847, 0.743474, 0.742098, 0.740720,
                  0.739339, 0.737954, 0.736567, 0.735176, 0.733783, 0.732387, 0.730989, 0.729588, 0.728184, 0.726778,
                  0.725369, 0.723957, 0.722544, 0.721128, 0.719710, 0.718291, 0.716870, 0.715446, 0.714020, 0.712593,
                  0.711164, 0.709732, 0.708299, 0.706864, 0.705427, 0.703988, 0.702548, 0.701107, 0.699665, 0.698221,
                  0.696775, 0.695327, 0.693879, 0.692429, 0.690977, 0.689525, 0.688071, 0.686617, 0.685161, 0.683705,
                  0.682248, 0.680789, 0.679330, 0.677869, 0.676408, 0.674947, 0.673485, 0.672023, 0.670560, 0.669097,
                  0.667634, 0.666171, 0.664707, 0.663243, 0.661778, 0.660313, 0.658849, 0.657384, 0.655919, 0.654455,
                  0.652991, 0.651527, 0.650064, 0.648601, 0.647138, 0.645675, 0.644213, 0.642752, 0.641291, 0.639831,
                  0.638371, 0.636911, 0.635452, 0.633992, 0.632534, 0.631076, 0.629619, 0.628163, 0.626708, 0.625254,
                  0.623801, 0.622350, 0.620899, 0.619450, 0.618001, 0.616554, 0.615108, 0.613662, 0.612218, 0.610776,
                  0.609334, 0.607894, 0.606455, 0.605018, 0.603582, 0.602148, 0.600715, 0.599284, 0.597855, 0.596428,
                  0.595002, 0.593578, 0.592156, 0.590737, 0.589319, 0.587903, 0.586490, 0.585079, 0.583669, 0.582261,
                  0.580856, 0.579452, 0.578050, 0.576650, 0.575252, 0.573856, 0.572462, 0.571071, 0.569682, 0.568296,
                  0.566911, 0.565530, 0.564151, 0.562774, 0.561400, 0.560029, 0.558660, 0.557295, 0.555932, 0.554571,
                  0.553213, 0.551857, 0.550504, 0.549154, 0.547806, 0.546462, 0.545120, 0.543782, 0.542446, 0.541114,
                  0.539785, 0.538458, 0.537135, 0.535816, 0.534499, 0.533186, 0.531875, 0.530567, 0.529261, 0.527959,
                  0.526661, 0.525366, 0.524074, 0.522785, 0.521500, 0.520217, 0.518938, 0.517663, 0.516392, 0.515123,
                  0.513858, 0.512596, 0.511338, 0.510083, 0.508832, 0.507585, 0.506341, 0.505100, 0.503865, 0.502632,
                  0.501403, 0.500178, 0.498956, 0.497738, 0.496525, 0.495315, 0.494108, 0.492905, 0.491706, 0.490511,
                  0.489320, 0.488132, 0.486947, 0.485768, 0.484592, 0.483419, 0.482250, 0.481085, 0.479925, 0.478769,
                  0.477616, 0.476467, 0.475322, 0.474182, 0.473046, 0.471913, 0.470785, 0.469662, 0.468543, 0.467428,
                  0.466316, 0.465210, 0.464108, 0.463010, 0.461915, 0.460826, 0.459740, 0.458659, 0.457581, 0.456509,
                  0.455440, 0.454375, 0.453315, 0.452260, 0.451208, 0.450160, 0.449117, 0.448079, 0.447044, 0.446013,
                  0.444987, 0.443966, 0.442948, 0.441934, 0.440926, 0.439921, 0.438920, 0.437926, 0.436935, 0.435947,
                  0.434965, 0.433987, 0.433013, 0.432043, 0.431078, 0.430117, 0.429160, 0.428209, 0.427261, 0.426317,
                  0.425380, 0.424446, 0.423515, 0.422591, 0.421670, 0.420752, 0.419840, 0.418932, 0.418027, 0.417128,
                  0.416233, 0.415341, 0.414455, 0.413573, 0.412694, 0.411821, 0.410953, 0.410087, 0.409228, 0.408372,
                  0.407520, 0.406674, 0.405831, 0.404993, 0.404159, 0.403328, 0.402503, 0.401682, 0.400863, 0.400051,
                  0.399242, 0.398437, 0.397637, 0.396840, 0.396049, 0.395262, 0.394478, 0.393700, 0.392925, 0.392155,
                  0.391389, 0.390626, 0.389870, 0.389116, 0.388368, 0.387623, 0.386882, 0.386147, 0.385414, 0.384686,
                  0.383962, 0.383242, 0.382527, 0.381815, 0.381109, 0.380405, 0.379706, 0.379012, 0.378320, 0.377635,
                  0.376951, 0.376273, 0.375598, 0.374927, 0.374261, 0.373598, 0.372941, 0.372285, 0.371636, 0.370989,
                  0.370347, 0.369709, 0.369074, 0.368444, 0.367817, 0.367195, 0.366576, 0.365962, 0.365350, 0.364745,
                  0.364141, 0.363543, 0.362947, 0.362356, 0.361768, 0.361184, 0.360604, 0.360027, 0.359455, 0.358886,
                  0.358321, 0.357759, 0.357202, 0.356647, 0.356097, 0.355550, 0.355007, 0.354468, 0.353932, 0.353400,
                  0.352872, 0.352346, 0.351825, 0.351307, 0.350793, 0.350282, 0.349775, 0.349271, 0.348770, 0.348274,
                  0.347779, 0.347290, 0.346802, 0.346320, 0.345838, 0.345363, 0.344889, 0.344421, 0.343953, 0.343491,
                  0.343031, 0.342574, 0.342121, 0.341670, 0.341224, 0.340779, 0.340340, 0.339901, 0.339468, 0.339036,
                  0.338608, 0.338184, 0.337761, 0.337344, 0.336926, 0.336516, 0.336105, 0.335699, 0.335296, 0.334894,
                  0.334498, 0.334101, 0.333711, 0.333322, 0.332936, 0.332554, 0.332171, 0.331796, 0.331421, 0.331050,
                  0.330682, 0.330313, 0.329952, 0.329591, 0.329232, 0.328878, 0.328524, 0.328176, 0.327828, 0.327483,
                  0.327143, 0.326802, 0.326466, 0.326132, 0.325798, 0.325471, 0.325144, 0.324819, 0.324499, 0.324178,
                  0.323861, 0.323548, 0.323234, 0.322925, 0.322619, 0.322311, 0.322011, 0.321711, 0.321411, 0.321118,
                  0.320824, 0.320532, 0.320245, 0.319959, 0.319673, 0.319393, 0.319113, 0.318834, 0.318561, 0.318288,
                  0.318015, 0.317749, 0.317483, 0.317216, 0.316957, 0.316697, 0.316437, 0.316184, 0.315931, 0.315678,
                  0.315429, 0.315183, 0.314937, 0.314693, 0.314454, 0.314214, 0.313975, 0.313742, 0.313510, 0.313277,
                  0.313049, 0.312823, 0.312598, 0.312371, 0.312153, 0.311935, 0.311716, 0.311500, 0.311288, 0.311077,
                  0.310864, 0.310658, 0.310454, 0.310249, 0.310044, 0.309846, 0.309648, 0.309450, 0.309252, 0.309062,
                  0.308872, 0.308682, 0.308490, 0.308308, 0.308125, 0.307942, 0.307758, 0.307581, 0.307406, 0.307230,
                  0.307053, 0.306881, 0.306713, 0.306545, 0.306376, 0.306207, 0.306047, 0.305886, 0.305725, 0.305563,
                  0.305405, 0.305252, 0.305099, 0.304945, 0.304791, 0.304642, 0.304497, 0.304351, 0.304205, 0.304058,
                  0.303917, 0.303780, 0.303642, 0.303504, 0.303365, 0.303229, 0.303100, 0.302971, 0.302841, 0.302711,
                  0.302580, 0.302457, 0.302336, 0.302215, 0.302094, 0.301973, 0.301852, 0.301738, 0.301626, 0.301515,
                  0.301403, 0.301291, 0.301180, 0.301071, 0.300970, 0.300868, 0.300767, 0.300665, 0.300564, 0.300463,
                  0.300366, 0.300275, 0.300185, 0.300095, 0.300004, 0.299914, 0.299825, 0.299736, 0.299653, 0.299574,
                  0.299495, 0.299417, 0.299339, 0.299261, 0.299184, 0.299107, 0.299030, 0.298962, 0.298896, 0.298830,
                  0.298764, 0.298699, 0.298635, 0.298570, 0.298506, 0.298443, 0.298380, 0.298323, 0.298270, 0.298218,
                  0.298168, 0.298119, 0.298070, 0.298023, 0.297977, 0.297932, 0.297887, 0.297844, 0.297800, 0.297757,
                  0.297714]  # diffusion
        y1k_df = [y - 0.07 for y in y1k_df]  # adjust value of y
        x01_rf = [x1k_df[0], x1k_df[-1]]  # rectified flow
        y01_rf = [y1k_df[0], y1k_df[-1]]  # rectified flow
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', labelsize=30)
        ax.plot(x1k_df, y1k_df, linestyle='-', color='r')
        ax.plot(x01_rf, y01_rf, linestyle='-', color='b', marker='o', ms=6)
        legends = ['Diffusion models trajectory', 'Rectified Flow trajectory']
        ax.legend(legends, fontsize=35, loc='upper right')
        ax.set_title(f"Trajectory: Diffusion models vs Rectified Flow", fontsize=40)
        ax.set_xlabel(r"$x_t[a]$", fontsize=35)
        ax.set_ylabel(r"$x_t[b]$        ", fontsize=35, rotation=0)
        ax.set_ylim((0.1, 1.0))
        text_kwargs = dict(ha='center', va='center', fontsize=40, color='k')
        # plt.text(0.25, 0.94, r"$\pi_1$", text_kwargs)
        # plt.text(0.64, 0.27, r"$\pi_0$", text_kwargs)
        fig.tight_layout()

        f_path = './configs/chart_dual_loss/fig_trajectory1_diffusion_vs_rectified_flow.png'
        fig.savefig(f_path, bbox_inches='tight')
        print(f"file saved: {f_path}")
        plt.close()

# class