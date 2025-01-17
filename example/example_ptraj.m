%%
clear all
ncload('/media/xzhou473/Seagate Backup Plus Drive/LLM_fish/fish_llm/example/trajectories_LLM.nc');
xLLM=x;
yLLM=y;
zLLM=z;
bathyLLM=bathy;
ncload('/media/xzhou473/Seagate Backup Plus Drive/LLM_fish/fish_llm/example/trajectories.nc');

%%
figure(11)
plot(squeeze(x(1,1,:)),squeeze(y(1,1,:)),'r--','linewidth',2.0);hold on
plot(squeeze(x(1,2,:)),squeeze(y(1,2,:)),'b--','linewidth',2.0);hold on
plot(squeeze(xLLM(1,1,:)),squeeze(yLLM(1,1,:)),'r','linewidth',2.0);hold on
plot(squeeze(xLLM(1,2,:)),squeeze(yLLM(1,2,:)),'b','linewidth',2.0);hold on
%%
figure(12)
plot([1:1:60],squeeze(zLLM(1,1,:)),'r','linewidth',2.0);hold on
plot([1:1:60],squeeze(zLLM(2,1,:)),'b','linewidth',2.0);hold on
plot([1:1:60],squeeze(bathyLLM(1,1,:)),'r--','linewidth',2.0);hold on
plot([1:1:60],squeeze(bathyLLM(1,2,:)),'b--','linewidth',2.0);hold on
%plot(squeeze(x(15,:,1)),squeeze(y(15,:,1)),'b','LineWidth',2.0);hold on

