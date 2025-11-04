%% ============================================================
% COMSOL LiveLink for MATLAB - CPL 批量参数样本仿真
% 说明：
% - 每个样本使用局部 meshParams，网格失败时放宽，仅在本样本内修改
% - 在写参数失败或几何/网格构建失败时跳过样本并记录日志
% ============================================================

import com.comsol.model.*
import com.comsol.model.util.*

rng(2025); % 固定随机种子

%% === 1. 参数基准与区间 ===
paramBase.core_y       = [10, 20];   % 允许区间（脚本会在每个样本中随机取值）
paramRanges.base_x     = [30, 50];
paramRanges.base_z     = [1, 4];
paramRanges.g_1        = [1.0, 2.0];
paramRanges.g_2        = [0.2, 1.0];
paramRanges.g_3        = [0.8, 1.5];
paramRanges.thick_copper = [0.04, 0.12];
paramRanges.thick_FR4    = [1.0, 2.0];
paramRanges.w_1        = [1.0, 3.0];
paramRanges.w_2        = [1.0, 3.0];
paramRanges.n          = [2, 6];       % integer
paramRanges.I          = [5, 15];

%% === 2. 智能网格控制参数 ===
initialMeshParams.d_mesh_min = 5400;
initialMeshParams.mesh_res = 0.5;
initialMeshParams.mesh_growth = 1.5;

% 网格 hmax 策略
baseHmaxFactor = 1/10;        % 初始 hmax = L * factor（比之前更保守）
hmaxIncreaseFactor = 2;       % 每次重试 hmax *= factor
maxMeshRetries = 4;           % 每个样本最多尝试次数
hmaxUpperLimitFactor = 5;     % hmax 最大不得超过 L * hmaxUpperLimitFactor

%% === 3. 样本数量与初始化文件 ===
nRandom = 5000;   
nExtreme = 1000;
nTotal = nRandom + nExtreme;

csvFile = 'results_CPL.csv';
headers = {'base_x','base_z','base_y','g_1','g_2','g_3','thick_copper','thick_FR4','thick_pcb','w_1','w_2','core_y','r','n','I','notes'};
fid = fopen(csvFile,'w');
if fid == -1
    error('无法创建 CSV 文件: %s', csvFile);
end
fprintf(fid,'%s\n',strjoin(headers,','));
fclose(fid);

logFile = 'failed_samples_CPL_log.txt';
if exist(logFile,'file'), delete(logFile); end

%% === 4. 打开模型 ===
modelFile = 'CPL.mph';
try
    model = mphopen(modelFile);
catch ME
    error('无法打开模型 %s: %s', modelFile, ME.message);
end

%% === 5. 主循环：生成样本 -> 写参数 -> 重建几何 -> mesh 重试 -> 运行 study -> 提取结果 ===
sampleCount = 0;
attempt = 0;
maxAttempts = nTotal * 10; % 上限，防止无限循环

while sampleCount < nTotal && attempt < maxAttempts
    attempt = attempt + 1;
    % 决定是否为极端样本（前 nExtreme 个为极端）
    isExtreme = (sampleCount < nExtreme);
    
    % 采样 params（注意 core_y 这里按区间取值）
    params = struct();
    % core_y 取值
    core_y_lim = paramBase.core_y;
    if isExtreme
        if rand < 0.5, params.core_y = core_y_lim(1); else params.core_y = core_y_lim(2); end
    else
        params.core_y = core_y_lim(1) + rand*(core_y_lim(2)-core_y_lim(1));
    end
    
    % 采样其它参数
    names_fixed = {'base_x','base_z','g_1','g_2','g_3','thick_copper','thick_FR4','w_1','w_2','n','I'};
    for k = 1:numel(names_fixed)
        nm = names_fixed{k};
        lim = paramRanges.(nm);
        if isExtreme
            if rand < 0.5
                val = lim(1);
            else
                val = lim(2);
            end
        else
            val = lim(1) + rand*(lim(2)-lim(1));
        end
        if strcmp(nm,'n')
            val = round(val);
            val = max(paramRanges.n(1), min(paramRanges.n(2), val));
        end
        params.(nm) = val;
    end
    
    % 计算 base_y 可行区间并采样
    lower_by = params.core_y + params.base_x/2 + 0.01;
    upper_by = min(35, params.core_y + params.base_x - 0.01);
    if upper_by <= lower_by
        fid = fopen(logFile,'a');
        fprintf(fid,'Skip: infeasible base_y range (base_x=%.6g) at attempt %d\n', params.base_x, attempt);
        fclose(fid);
        continue;
    end
    if isExtreme
        if rand < 0.5, params.base_y = lower_by; else params.base_y = upper_by; end
    else
        params.base_y = lower_by + rand*(upper_by - lower_by);
    end
    
    % 计算 thick_pcb（脚本同时写入该数值到模型，若希望模型内用表达式可注释掉）
    params.thick_pcb = params.n * params.thick_copper + (params.n - 1) * params.thick_FR4;
    
    % 计算 r 上界并采样
    ub1 = params.base_x/4 - params.w_1/2;
    ub2 = (params.base_y - params.core_y)/2;
    r_ub = min(ub1, ub2) - 1e-3;
    if r_ub <= 0
        fid = fopen(logFile,'a');
        fprintf(fid,'Skip: infeasible r upper bound (ub1=%.6g, ub2=%.6g) at attempt %d\n', ub1, ub2, attempt);
        fclose(fid);
        continue;
    end
    if isExtreme
        if rand < 0.5
            params.r = max(0.05, r_ub*0.9);
        else
            params.r = max(0.05, r_ub*0.3);
        end
    else
        params.r = 0.05 + rand*(r_ub - 0.05);
    end
    
    % 基本几何检查
    if (params.base_x/4 - params.w_1/2) <= 0
        fid = fopen(logFile,'a');
        fprintf(fid,'Skip: base_x too small relative to w_1 (base_x=%.6g, w_1=%.6g) at attempt %d\n', params.base_x, params.w_1, attempt);
        fclose(fid);
        continue;
    end
    
    % 准备写参数并运行
    sampleCount = sampleCount + 1;
    fprintf('\n===== CPL Sample %d/%d (attempt %d) =====\n', sampleCount, nTotal, attempt);
    
    %% ---- 6. 稳健写入模型参数（n 以整数写入） ----
    allNames = fieldnames(params);
    failedSet = false;
    for p = 1:length(allNames)
        nm = allNames{p};
        val = params.(nm);
        try
            if strcmp(nm,'n')
                model.param.set(nm, sprintf('%d', round(val)));
            else
                model.param.set(nm, sprintf('%.12g', val));
            end
        catch ME
            warning('无法写入参数 %s 到模型: %s', nm, ME.message);
            fid = fopen(logFile,'a');
            fprintf(fid,'Failed to set parameter %s at sample %d (attempt %d): %s\n', nm, sampleCount, attempt, ME.message);
            fclose(fid);
            failedSet = true;
            break;
        end
    end
    if failedSet
        fprintf('跳过样本（参数写入失败），已记录到 %s\n', logFile);
        continue;
    end
    
    %% ---- 7. 强制重建几何（确保新参数生效） ----
    try
        compTags = model.component.tags;
        if ~isempty(compTags)
            comp = char(compTags(1));
            geomTags = model.component(comp).geom.tags;
            for g = 1:numel(geomTags)
                try
                    model.component(comp).geom(char(geomTags(g))).run;
                catch geomErr
                    error('Geometry run error: %s', geomErr.message);
                end
            end
        end
    catch ME
        warning('几何重建失败: %s', ME.message);
        fid = fopen(logFile,'a');
        fprintf(fid,'Geometry rebuild failed at sample %d (attempt %d): %s\n', sampleCount, attempt, ME.message);
        fclose(fid);
        continue;
    end
    
    %% ---- 8. 局部 meshParams 与多级尝试 ----
    meshParams = initialMeshParams; % 每样本独立
    % 估算特征长度 L（这里用 base_x 或 base_y 之一作为尺度）
    L = max(abs(params.base_x), abs(params.base_y));
    if L <= 0, L = 1; end
    initial_hmax = max(L * baseHmaxFactor, 1e-6);
    hmax = initial_hmax;
    hmax_upper = L * hmaxUpperLimitFactor;
    meshSuccess = false;
    tried_hmax_list = [];
    
    % 获取 comp/mesh tags 以便后续运行
    compTags = model.component.tags;
    if isempty(compTags)
        warning('模型没有 component 标签，跳过样本');
        fid = fopen(logFile,'a'); fprintf(fid,'No component tags at sample %d\n', sampleCount); fclose(fid);
        continue;
    end
    comp = char(compTags(1));
    meshTags = model.component(comp).mesh.tags;
    
    for retry = 1:maxMeshRetries
        tried_hmax_list(end+1) = hmax; %#ok<SAGROW>
        % 1) 写入可能存在的 mesh size features hmax/hmin（若 feature 名含 size）
        try
            for m = 1:numel(meshTags)
                mtag = char(meshTags(m));
                meshObj = model.component(comp).mesh(mtag);
                % 查找 feature tags
                try
                    ftags = meshObj.feature.tags;
                    for ft = 1:numel(ftags)
                        ftname = char(ftags(ft));
                        if contains(lower(ftname),'size')
                            % 尝试设置 hmax/hmin（某些模型可能使用参数名不同，故用 try/catch）
                            try meshObj.feature(ftname).set('hmax', sprintf('%.12g', hmax)); catch; end
                            try meshObj.feature(ftname).set('hmin', sprintf('%.12g', hmax/10)); catch; end
                        end
                    end
                catch; end
            end
        catch; end
        
        % 2) 将本次 meshParams 写入 model.param（以防模型使用这些参数）
        try
            model.param.set('d_mesh_min', sprintf('%.12g', meshParams.d_mesh_min));
            model.param.set('mesh_res', sprintf('%.12g', meshParams.mesh_res));
            model.param.set('mesh_growth', sprintf('%.12g', meshParams.mesh_growth));
        catch; end
        
        % 3) 运行 mesh（逐个 mesh tag）
        try
            for m = 1:numel(meshTags)
                try
                    model.component(comp).mesh(char(meshTags(m))).run;
                catch meshRunErr
                    error('Mesh run error (tag=%s): %s', char(meshTags(m)), meshRunErr.message);
                end
            end
            meshSuccess = true;
            if retry == 1
                fprintf('  mesh: 初始 hmax=%.4g 成功\n', hmax);
            else
                fprintf('  mesh: 第 %d 次放宽后成功（hmax=%.4g）\n', retry, hmax);
            end
            break;
        catch meshErr
            % 捕获 mesh 错误并放宽策略（仅修改本样本的 meshParams）
            fprintf('  mesh: 尝试 %d 失败 (hmax=%.4g). 错误（摘要）: %s\n', retry, hmax, meshErr.message(1:min(200,end)));
            hmax = min(hmax * hmaxIncreaseFactor, hmax_upper);
            meshParams.d_mesh_min = meshParams.d_mesh_min * 1.5;
            meshParams.mesh_res = meshParams.mesh_res * 1.2;
            meshParams.mesh_growth = meshParams.mesh_growth * 1.1;
            pause(0.08);
        end
    end
    
    if ~meshSuccess
        warning('⚠️ CPL 样本 %d: 多级网格尝试失败，跳过该样本（尝试的 hmax：%s）', sampleCount, mat2str(tried_hmax_list));
        % 记录日志
        try
            fid = fopen(logFile,'a');
            fprintf(fid,'====== CPL Sample %d mesh failed ======\n', sampleCount);
            for kf = 1:length(allNames)
                fprintf(fid,'%s = %.12g\n', allNames{kf}, params.(allNames{kf}));
            end
            fprintf(fid,'hmax_tried = %s\n', mat2str(tried_hmax_list));
            fprintf(fid,'meshParams_final: d_mesh_min=%.12g, mesh_res=%.12g, mesh_growth=%.12g\n\n', meshParams.d_mesh_min, meshParams.mesh_res, meshParams.mesh_growth);
            fclose(fid);
        catch; end
        continue;
    end
    
    pause(0.05);
    
    %% ---- 9. 运行 study ----
    try
        try
            model.study('std1').runAll;
        catch
            model.study('std1').run;
        end
    catch ME
        warning('❌ CPL 样本 %d 仿真失败: %s', sampleCount, ME.message);
        try fname = sprintf('CPL_failed_sample_%04d.mph', sampleCount); model.save(fname); end
        try
            fid = fopen(logFile,'a');
            fprintf(fid,'====== CPL Sample %d sim failed ======\n', sampleCount);
            for kk = 1:length(allNames)
                fprintf(fid,'%s = %.12g\n', allNames{kk}, params.(allNames{kk}));
            end
            fprintf(fid,'Error: %s\n\n', ME.message);
            fclose(fid);
        catch; end
        continue;
    end
    
%% ---- 10. 提取结果 ----
% 提取结果：
%  - mf.LCoil_winding_left (uH)
%  - mf.L_winding_right_winding_left (uH)
%  - CPL_ripple
%  - CPL_volume
%  - ht.QInt (mW)
%  - ht.hf1.Tave (℃)

LCoil_winding_left_uH = NaN;
L_winding_right_winding_left_uH = NaN;
CPL_ripple = NaN;
CPL_volume = NaN;
ht_QInt_mW = NaN;
ht_hf1_Tave_C = NaN;
notes = '';

try
    val = mphglobal(model,'mf.LCoil_winding_left');
    if ~isempty(val), LCoil_winding_left_uH = val * 1e6; end
    notes = [notes, sprintf('LCoil_left=%.6g uH; ', LCoil_winding_left_uH)];
catch ME
    notes = [notes, sprintf('err:LCoil(%s); ', ME.message(1:min(120,end)))];
end

try
    val = mphglobal(model,'mf.L_winding_right_winding_left');
    if ~isempty(val), L_winding_right_winding_left_uH = val * 1e6; end
    notes = [notes, sprintf('Lmut=%.6g uH; ', L_winding_right_winding_left_uH)];
catch ME
    notes = [notes, sprintf('err:Lmut(%s); ', ME.message(1:min(120,end)))];
end

try
    val = mphglobal(model,'CPL_ripple');
    if ~isempty(val), CPL_ripple = val; end
    notes = [notes, sprintf('ripple=%.6g; ', CPL_ripple)];
catch ME
    notes = [notes, sprintf('err:ripple(%s); ', ME.message(1:min(120,end)))];
end

try
    val = mphglobal(model,'CPL_volume');
    if ~isempty(val), CPL_volume = val; end
    notes = [notes, sprintf('volume=%.6g; ', CPL_volume)];
catch ME
    notes = [notes, sprintf('err:volume(%s); ', ME.message(1:min(120,end)))];
end

try
    val = mphglobal(model,'ht.QInt');
    if ~isempty(val), ht_QInt_mW = val * 1e3; end  % W → mW
    notes = [notes, sprintf('QInt=%.6g mW; ', ht_QInt_mW)];
catch ME
    notes = [notes, sprintf('err:QInt(%s); ', ME.message(1:min(120,end)))];
end

try
    val = mphglobal(model,'ht.hf1.Tave');
    if ~isempty(val), ht_hf1_Tave_C = val - 273.15; end
    notes = [notes, sprintf('Tave=%.3g C; ', ht_hf1_Tave_C)];
catch ME
    notes = [notes, sprintf('err:Tave(%s); ', ME.message(1:min(120,end)))];
end

    
    %% ---- 11. 写入 CSV ----
    dataRow = [params.base_x, params.base_z, params.base_y, params.g_1, params.g_2, params.g_3, ...
               params.thick_copper, params.thick_FR4, params.thick_pcb, params.w_1, params.w_2, ...
               params.core_y, params.r, params.n, params.I];
    fid = fopen(csvFile,'a');
    fprintf(fid, [repmat('%.12g,',1,length(dataRow)) '%s\n'], dataRow, notes);
    fclose(fid);
    
    fprintf('✅ CPL 样本 %d 已完成并写入 %s\n', sampleCount, csvFile);
end

if sampleCount < nTotal
    warning('样本生成停止：仅生成 %d/%d 个有效样本（达到最大尝试次数 %d）', sampleCount, nTotal, maxAttempts);
else
    disp('所有样本生成并仿真完成');
end
