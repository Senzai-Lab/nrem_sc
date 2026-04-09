% Extracting sleep states into separate arrays
data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\107b\";
processed_data_dir = data_dir + "processed\107b\";

filename = "YutaTest116b.SleepState.states.mat";
load(raw_data_dir + filename);

wake = SleepState.ints.WAKEstate;
nrem = SleepState.ints.NREMstate;
rem = SleepState.ints.REMstate;

save(processed_data_dir + "wake.mat", 'wake');
save(processed_data_dir + "nrem.mat", 'nrem');
save(processed_data_dir + "rem.mat", 'rem');

% Extracting neck position
filename = "YutaTest116b_OpenField_HeadDirection.mat";
load(raw_data_dir + filename);

x = NeckPosX(~NeckOmitIdx);
y = NeckPosY(~NeckOmitIdx);
t_new = t(~NeckOmitIdx);
pos = [x, y];

save(processed_data_dir + "neck_pos.mat", 'pos', '-double');
save(processed_data_dir + "neck_time", 't_new', '-double');

% Extracting turn cells from SC (shank 3)
filename = 'MetaAnalysis_20241220-metaUnitFeature.mat';
load("D:\common_datasets\ucsf\raw\" + filename);

mouseID = metaUnitFeature.mouseID==6; % Mouse 107b
turnModCells = metaUnitFeature.SC_TurnModCell==1; % Select turn modulated cells
shankID = metaUnitFeature.shankID==3; % Shank 3 in SC
turn_ids = metaUnitFeature.unitIDshank(mouseID & turnModCells & shankID);

save(processed_data_dir + "turn_ids", "turn_ids");
% metaUnitFeature.TurnIdx contains turn modulation strength

% Extract HD cells from ADN (shank 1 and 2)
filename = 'MetaAnalysis_20241220-metaUnitFeature.mat';
load(raw_data_dir + filename);

mouseID = metaUnitFeature.mouseID==6; % Mouse 107b
hd_cells= metaUnitFeature.HDCidx==1; % Select head direction tuned cells
shankID = (metaUnitFeature.shankID==1) | (metaUnitFeature.shankID==2); %     ADN
hd_ids = metaUnitFeature.unitIDshank(mouseID & hd_cells & shankID);

save(processed_data_dir + "hd_ids", "hd_ids");

% CW and CCW turn cell IDs
Idx_modCW = metaUnitFeature.Idx_modCW(mouseID & turnModCells & shankID);
Idx_modCCW = metaUnitFeature.Idx_modCCW(mouseID & turnModCells & shankID);
id_ccw = (Idx_modCW - Idx_modCCW) < 0;
mod_idx = [Idx_modCW, Idx_modCW];

save(processed_data_dir + 'mod_idx.mat', 'mod_idx');
save(processed_data_dir + "id_ccw.mat", 'id_ccw');

% Unit 116b shank info
filename = 'YutaTest116b_BayesianDecoding_training.mat';
load(raw_data_dir + filename);
save(processed_data_dir + 'unit_idx.mat', 'unit_idx');
% 98 - 70 - 117 - 106
shank1 = unit_idx(1:98);
shank2 = unit_idx(99:168);
shank3 = unit_idx(169:285);
shank4 = unit_idx(286:end);


% Unit 85b HD indices
data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\85b\";
processed_data_dir = data_dir + "processed\85b\";

% Extract HD indices
load(raw_data_dir + 'depthsort_parameter_1.mat');
shank1_len = length(depth); % 17

load(raw_data_dir + 'depthsort_parameter_2.mat');
shank2_len = length(depth); % 80
% first 17 units (shank1)
% later 80 units (shank2)

filename = 'YutaTest85b_BayesianDecoding_training.mat';
load(raw_data_dir + filename);
shank1 = unit_idx(1:shank1_len);
shank2 = unit_idx(shank1_len+1:end);

save(processed_data_dir + 'shank1_hd_idx', 'shank1');
save(processed_data_dir + 'shank2_hd_idx', 'shank2');


% Sleep state
load(raw_data_dir + 'YutaTest85b.SleepState.states.mat')
wake = SleepState.ints.WAKEstate;
nrem = SleepState.ints.NREMstate;
rem = SleepState.ints.REMstate;
qwake = SleepState.ints.QuietWakestate;

% Merge info
load(raw_data_dir + 'YutaTest85b_merge_info.mat');
session_epochs = merge_info.sessionperiod;
save(processed_data_dir + 'session_epochs', 'session_epochs');

% 83b
data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\83b\";
processed_data_dir = data_dir + "processed\83b\";

filename = 'YutaTest83b_BayesianDecoding_training.mat';
load(raw_data_dir + filename);
shank1 = unit_idx(1:shank1_len);
shank2 = unit_idx(shank1_len+1:end);
save(processed_data_dir + 'unit_idx', 'unit_idx');

% Sleep state
load(raw_data_dir + 'YutaTest83b.SleepState.states.mat')
wake = SleepState.ints.WAKEstate;
nrem = SleepState.ints.NREMstate;
rem = SleepState.ints.REMstate;
qwake = SleepState.ints.quietwakestate;

save(processed_data_dir + "wake.mat", 'wake');
save(processed_data_dir + "nrem.mat", 'nrem');
save(processed_data_dir + "rem.mat", 'rem');
save(processed_data_dir + "qwake.mat", 'qwake');

% Merge info
load(raw_data_dir + 'YutaTest83b_merge_info.mat');
session_epochs = merge_info.sessionperiod;
save(processed_data_dir + 'session_epochs', 'session_epochs');

% 119b
data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\119b\";
processed_data_dir = data_dir + "processed\119b\";

d1 = load("depthsort_parameter_1.mat").depth;
d2 = load("depthsort_parameter_2.mat").depth;
d3 = load("depthsort_parameter_3.mat").depth;
d4 = load("depthsort_parameter_4.mat").depth;

filename = 'YutaTest119b_BayesianDecoding_training.mat';
load(raw_data_dir + filename);
save(processed_data_dir + 'unit_idx', 'unit_idx');

% Sleep state
load(raw_data_dir + 'YutaTest119b.SleepState.states.mat')
wake = SleepState.ints.WAKEstate;
nrem = SleepState.ints.NREMstate;
rem = SleepState.ints.REMstate;

save(processed_data_dir + "wake.mat", 'wake');
save(processed_data_dir + "nrem.mat", 'nrem');
save(processed_data_dir + "rem.mat", 'rem');

% Merge info
load(raw_data_dir + 'YutaTest119b_merge_info.mat');
session_epochs = merge_info.sessionperiod;
save(processed_data_dir + 'session_epochs', 'session_epochs');