% Extracting sleep states into separate arrays
data_dir = '/Users/iii9781/nrem_sc/data/';
raw_data_dir = data_dir + "raw/";
processed_data_dir = data_dir + "processed/";

filename = "YutaTest107b.SleepState.states.mat";
load(raw_data_dir + filename);

wake = SleepState.ints.WAKEstate;
nrem = SleepState.ints.NREMstate;
rem = SleepState.ints.REMstate;

save(processed_data_dir + "wake.mat", 'wake');
save(processed_data_dir + "nrem.mat", 'nrem');
save(processed_data_dir + "rem.mat", 'rem');

% Extracting neck position
filename = "YutaTest107b_OpenField_HeadDirection.mat";
load(raw_data_dir + filename);

x = NeckPosX(~NeckOmitIdx);
y = NeckPosY(~NeckOmitIdx);
t_new = t(~NeckOmitIdx);
pos = [x, y];

save(processed_data_dir + "neck_pos.mat", 'pos', '-double');
save(processed_data_dir + "neck_time", 't_new', '-double');

% Extracting turn cells from SC
filename = 'MetaAnalysis_20241220-metaUnitFeature.mat';
load(raw_data_dir + filename);

mouseID = metaUnitFeature.mouseID==6; % Mouse 107b
turnModCells = metaUnitFeature.SC_TurnModCell==1; % Select turn modulated cells
shankID = metaUnitFeature.shankID==3; % Shank 3 in SC
turn_ids = metaUnitFeature.unitIDshank(mouseID & turnModCells & shankID);

save(processed_data_dir + "turn_ids", "turn_ids");

% CW and CCW turn cell IDs
Idx_modCW = metaUnitFeature.Idx_modCW(mouseID & turnModCells & shankID);
Idx_modCCW = metaUnitFeature.Idx_modCCW(mouseID & turnModCells & shankID);
id_ccw = (Idx_modCW - Idx_modCCW) < 0;

save(processed_data_dir + "id_ccw.mat", 'id_ccw');