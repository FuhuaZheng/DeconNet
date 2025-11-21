clear; clc; close all;
addpath('./src');
addpath('./src/ray_tracing')
addpath('./cvx')
%%
max_misfit = 0.3;
inputdir = './results/3_filtered_data/'; 
savepath = sprintf('./results/4_pics_2nd_results/', max_misfit);

if ~exist(savepath,'dir'); mkdir(savepath); end

catalogFile = './catalog/Events_match.txt';
fid = fopen(catalogFile, 'r');
catalogData = textscan(fid, '%s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
fclose(fid);
catalogData = table(catalogData{:}, 'VariableNames', {'Index', 'Target_ID', 'Target_Lat', 'Target_Lon', 'Target_Depth', 'Target_Mag', 'EGF_ID', 'EGF_Lat', 'EGF_Lon', 'EGF_Depth', 'EGF_Mag', 'Stk', 'Dip', 'Rake'});

allCatalogFile = './catalog/catalog.txt';
fid = fopen(allCatalogFile, 'r');
allCatalogData = textscan(fid, '%s %f %f %f %f %s %s %s %s', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
fclose(fid);
allLatitudes = allCatalogData{2};
allLongitudes = allCatalogData{3};
allMagnitudes = allCatalogData{5};

matFiles = dir(fullfile(inputdir, 'target_*.mat')); 
fprintf('Total .mat files found: %d\n', length(matFiles));
rand_idx = randperm(length(matFiles));
matFiles = matFiles(rand_idx);

Fs = 500;
[top_mod,vp_mod, vs_mod, ~, ~, ~]=textread('./Code_2nd_Seismic_Moment/toc2me_locateV.txt','%f %f %f %f %f %f');

processedFiles = 0;
magnitudes = zeros(length(matFiles), 1);
for fileIdx = 1:length(matFiles)
    if processedFiles >= 10
        break;
    end
    currentFile = fullfile(matFiles(fileIdx).folder, matFiles(fileIdx).name);
    [~, filename, ~] = fileparts(matFiles(fileIdx).name);
    evt_name = filename;
    parts = strsplit(evt_name, '_');
    evt_num = parts{2};
    evt_id = parts{3};
    egf_num = parts{4}(4:end);
    matchedIdx = strcmp(catalogData.Target_ID, evt_id);
    if any(matchedIdx)
        magnitudes(fileIdx) = str2double(catalogData.Target_Mag(matchedIdx));
        depths(fileIdx) = str2double(catalogData.Target_Depth(matchedIdx));
    else
        magnitudes(fileIdx) = NaN;
        depths(fileIdx) = NaN;
    end
    fprintf('Processing file %d: %s (Event ID: %s)\n', processedFiles + 1, matFiles(fileIdx).name, evt_id);
    load(currentFile);
    valid_index = find(misfits_512 < max_misfit);
    sta_misfit_512 = sta_S(valid_index);
    stla_misfit_512 = stla_S(valid_index);
    stlo_misfit_512 = stlo_S(valid_index);
    astf_nn_misfit_512 = astf_nn_S(valid_index, :);
    sta_misftt_512 = sta_S(valid_index);
    astf_nn_S = astf_nn_misfit_512';
    phas = repmat('S', length(stla_misfit_512), 1); 
    Tauc_S = zeros(length(stla_misfit_512), 1);
    for i = 1:length(Tauc_S)
        [Tauc_S(i),~] = Tauc_Tc_moment(astf_nn_S(:,i),1/Fs);
    end
    
    %% 二阶矩反演
    [toa_S,az_S,slowV_S] = get_slowness_2nd(evdp,evla,evlo,stla_misfit_512,stlo_misfit_512,vp_mod,vs_mod,top_mod,phas);  % 使用筛选后的台站位置
    fprintf('shape of toa_S: %s\n, shape of az_S: %s\n, shape of slowV_S: %s\n', mat2str(size(toa_S)), mat2str(size(az_S)), mat2str(size(slowV_S)));
    
    [L_c,W_c,dir,Drp,tauc,v0,mv0,Tauc_S_2nd_pre,~] = inversion_second_moment(toa_S,az_S,Tauc_S',slowV_S,stk,dip);

    fprintf('shape of Tauc_S_2nd_pre: %s\n', mat2str(size(Tauc_S_2nd_pre)));
    fprintf('shape of Tauc_S: %s\n', mat2str(size(Tauc_S)));
    fprintf('shape of tauc: %s\n', mat2str(size(tauc)));
    v0_east = v0(1)*sin(stk/180*pi) + v0(2)*cos(dip/180*pi)*cos(stk/180*pi);
    v0_north = v0(1)*cos(stk/180*pi) - v0(2)*cos(dip/180*pi)*sin(stk/180*pi);
    v0_up = -v0(2)*sin(dip/180*pi);
    rupture_angle = rad2deg(atan2(v0_east,v0_north));
    if rupture_angle < 0
        rupture_angle = rupture_angle+360;
    end
    display(rupture_angle);
    % results: L_c, W_c, tauc, mv0, dir, rupture angle  
    %% 绘图
    f = figure('visible', 'off');
    set(gcf,'PaperPositionMode','auto');
    set(gcf,'Units','inches');
    afFigurePosition = [0 0 22 10];
    set(gcf,'Position',afFigurePosition);
    set(gcf,'PaperSize',[afFigurePosition(1)*2+afFigurePosition(3) afFigurePosition(2)*2+afFigurePosition(4)]);
    tauc_range = [min(Tauc_S),max(Tauc_S)];

    % NN output τc map
    subplot(2,4,1)
    for i = 1:length(allLatitudes)
        scatter(allLongitudes(i), allLatitudes(i), 0.5*(3^allMagnitudes(i)), 'filled', 'MarkerFaceColor', [0.8 0.8 0.8], 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 1); hold on
    end
    for i = 1:length(az_S)
        scatter(stlo_misfit_512(i), stla_misfit_512(i), 20, Tauc_S(i), 'filled'); hold on
    end
    plot(evlo, evla, 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r')
    colormap turbo; colorbar; box on; axis square
    caxis(tauc_range)
    title('NN output \tau_c (sec)')
    ylabel('Latitude')
    xlabel('Longitude')
    set(gca, 'fontsize', 14)
    
    % NN output polar plot
    pax = subplot(2,4,5,polaraxes);
    for i = 1:length(az_S)
        polarscatter(az_S(i)/180*pi,toa_S(i),30,Tauc_S(i),'filled','linewidth',2.5); hold on
    end
    pax = gca;
    pax.ThetaDir = 'clockwise';
    pax.ThetaZeroLocation = 'top';
    colormap turbo;
    cb = colorbar;
    caxis(tauc_range)
    thetaticks(0:30:330);
    labelnm = get(gca,'thetaticklabel');
    set(gca,'thetaticklabel',strcat(labelnm(:),'^\circ'));
    rlimnm = get(gca,'rticklabel');
    set(gca,'rticklabel',strcat(rlimnm(:),'^\circ'));
    title('NN output \tau_c (sec)')
    set(gca,'fontsize',14)
    
    % 2nd moment prediction map
    subplot(2,4,2)
    for i = 1:length(allLatitudes)
        scatter(allLongitudes(i), allLatitudes(i), 0.5*(3^allMagnitudes(i)), 'filled', 'MarkerFaceColor', [0.8 0.8 0.8], 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 1); hold on
    end    
    for i = 1:length(az_S)
        scatter(stlo_misfit_512(i), stla_misfit_512(i), 20, Tauc_S_2nd_pre(i), 'filled'); hold on
    end
    plot(evlo, evla, 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r')
    quiver(evlo, evla, v0_east/8/cos(evla/180*pi), v0_north/8, 0.05, 'k', 'maxheadsize', 0.3, 'linewidth', 2)
    colormap turbo; colorbar; box on; axis square
    caxis(tauc_range)
    title('2nd Moment \tau_c (sec)')
    ylabel('Latitude')
    xlabel('Longitude')
    set(gca, 'fontsize', 14)
    
    % 2nd moment prediction polar plot
    pax = subplot(2,4,6,polaraxes);
    for i = 1:length(az_S)
        polarscatter(az_S(i)/180*pi,toa_S(i),30,Tauc_S_2nd_pre(i),'filled','linewidth',2.5); hold on
    end
    pax = gca;
    pax.ThetaDir = 'clockwise';
    pax.ThetaZeroLocation = 'top';
    colormap turbo;
    cb = colorbar;
    caxis(tauc_range)
    thetaticks(0:30:330);
    labelnm = get(gca,'thetaticklabel');
    set(gca,'thetaticklabel',strcat(labelnm(:),'^\circ'));
    rlimnm = get(gca,'rticklabel');
    set(gca,'rticklabel',strcat(rlimnm(:),'^\circ'));
    title('2nd Moment \tau_c (sec)')
    set(gca,'fontsize',14)
    
    % Tauc_S comparison plot
    subplot(2,4,[3,7])
    fprintf('shape of Tauc_S: %s, shape of az_S: %s\n', mat2str(size(Tauc_S)), mat2str(size(az_S)));
    scatter(Tauc_S, az_S, 30, 'd', 'MarkerEdgeColor', [0.4660, 0.6740, 0.1880], 'MarkerFaceColor', [0.4660, 0.6740, 0.1880]); hold on
    scatter(Tauc_S_2nd_pre, az_S, 30, '^', 'MarkerEdgeColor', [0.8500, 0.3250, 0.0980], 'MarkerFaceColor', [0.8500, 0.3250, 0.0980]);
    
    xlabel('S ASTFs \tau_c (sec)')
    ylabel('Azimuth')
    legend('NN out.','2nd Moment', 'Location', 'NorthWest')
    ylim([0, 380])
    % xlim([0.018 0.055])
    xlim([min(tauc_range)*0.85, max(tauc_range)*1.15])
    set(gca,'fontsize',15)
    box on
    title('Apparent \tau_c')
    
    subplot(2,4,[4,8])
    plot([0 0], [0 360], '--', 'Color', '#d9d9d9', 'LineWidth', 1.5, 'HandleVisibility', 'off'); hold on
    misfit = (Tauc_S_2nd_pre - Tauc_S) ./ Tauc_S * 100;
    scatter(misfit, az_S, 30, [0.8500, 0.3250, 0.0980], 'filled');
    fprintf('shape of misfit: %s\n', mat2str(size(misfit)));
    
    max_misfit_plot = max(abs(misfit)) * 1.1;
    for i = 1:length(sta_misfit_512)
        station_name = num2str(sta_misfit_512(i));
        % fprintf('Station name: %s\n', station_name);
        text(25 * 0.8, az_S(i), station_name, 'FontSize', 8, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    end

    text(25 * 0.795, 365, 'Sta', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    
    xlabel('\tau_c Misfit (%)') 
    ylabel('Azimuth')
    ylim([0, 380])
    xlim([-25, 25])
    set(gca,'fontsize',15)
    box on    
    misfit_mean = mean((Tauc_S_2nd_pre' - Tauc_S') ./ Tauc_S' * 100);
    title(['2nd Moment vs NN Misfit: ' sprintf('%+.2f', misfit_mean) '%'], 'FontSize', 15)

    year = evt_id(1:4);
    month = evt_id(5:6);
    day = evt_id(7:8);
    hour = evt_id(9:10);
    minute = evt_id(11:12);
    second = evt_id(13:14);
    millisecond = evt_id(16:end);  

    formatted_datetime = sprintf('%s/%s/%s %s:%s:%s', year, month, day, hour, minute, second);

    param_text = sprintf('2nd Moment Results: L_c: %.1f m, W_c: %.1f m, Tauc: %.3f s, Dir ratio: %.2f, V0: %.2f km/s, Rup direction: %.0f°', ...
        L_c*1000, W_c*1000, tauc, dir, mv0, rupture_angle);

    sgtitle({[sprintf('Inversion Results using S-ASTFs of Event %s, M = %.2f', formatted_datetime, magnitudes(fileIdx))], ...
            [sprintf('Stations used: %d/64 (misfits < %.1f)', length(sta_misfit_512), max_misfit)], ...
            param_text}, ...
            'FontSize', 18, 'FontWeight', 'bold');

    sv = fullfile(savepath, sprintf('target_%s_M%.2f_%s_egf_%s_2ndinv', evt_num, magnitudes(fileIdx), evt_id, egf_num));
    saveas(f, [sv, '.png']);
    close(f);
    
    processedFiles = processedFiles + 1;
end

fprintf('Processing completed, %d files processed\n', processedFiles);