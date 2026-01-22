% All data extracted from Mouse 107b
% Extracting sleep states into separate arrays
data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\116b\";
processed_data_dir = data_dir + "processed\116b\";

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
load(raw_data_dir + filename);

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
shankID = (metaUnitFeature.shankID==1) | (metaUnitFeature.shankID==2); % ADN
hd_ids = metaUnitFeature.unitIDshank(mouseID & hd_cells & shankID);

save(processed_data_dir + "hd_ids", "hd_ids");

% CW and CCW turn cell IDs
Idx_modCW = metaUnitFeature.Idx_modCW(mouseID & turnModCells & shankID);
Idx_modCCW = metaUnitFeature.Idx_modCCW(mouseID & turnModCells & shankID);
id_ccw = (Idx_modCW - Idx_modCCW) < 0;

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