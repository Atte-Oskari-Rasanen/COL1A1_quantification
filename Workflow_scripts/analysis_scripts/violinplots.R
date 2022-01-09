library("dplyr")
library("ggpubr")
library("onewaytests")
df1 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_data.csv")
df1$Animal_ID <- as.character(as.character(df1$Animal_ID)) 

df_coloc <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/group1_Ncoloc.csv") 
df_A_col1a1_A_hunu <- read.csv('/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group1/Group1_Acol1a1_Ahunu.csv')


df2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_data.csv")
df2$Animal_ID <- as.character(as.character(df2$Animal_ID)) 

df_coloc2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Ncoloc.csv") 
df_tot2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Nhunu.csv") 
df_A_coloc_N_hunu2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Acoloc_Nhunu.csv") 
df_N_coloc_A_hunu2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Ncoloc_Ahunu.csv") 
df_N_coloc_N_hunu2 <- read.csv("/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Ncoloc_Nhunu.csv") 
df_A_col1a1_A_hunu2 <- read.csv('/home/atte/Documents/PD_images/batch8_retry/datas_known_ids/group2/group2_Acol1a1_Ahunu.csv')

# df_area2[df_area2 == 0] <- NA
# df_coloc2[df_coloc2 == 0] <- NA
# df_tot2[df_tot2 == 0] <- NA
# df_A_coloc_N_hunu2[df_A_coloc_N_hunu2==0]<-NA
# df_N_coloc_A_hunu2[df_N_coloc_A_hunu2==0]<-NA
# df_N_coloc_N_hunu2[df_N_coloc_N_hunu2==0]<-NA
# df_A_col1a1_A_hunu2[df_A_col1a1_A_hunu2==0]<-NA

df_area_all <- data.frame(stats = c(t(df_area)), stringsAsFactors=FALSE)
df_coloc_all <- data.frame(stats = c(t(df_coloc)), stringsAsFactors=FALSE)
df_total_all <- data.frame(stats = c(t(df_tot)), stringsAsFactors=FALSE)
df_A_coloc_N_hunu_all <- data.frame(stats = c(t(df_A_coloc_N_hunu)), stringsAsFactors=FALSE)
df_N_coloc_A_hunu_all <- data.frame(stats = c(t(df_N_coloc_A_hunu)), stringsAsFactors=FALSE)
df_N_coloc_N_hunu_all <- data.frame(stats = c(t(df_N_coloc_N_hunu)), stringsAsFactors=FALSE)
df_A_col1a1_A_hunu_all <- data.frame(stats = c(t(df_A_col1a1_A_hunu)), stringsAsFactors=FALSE)

df_area_all2 <- data.frame(stats = c(t(df_area2)), stringsAsFactors=FALSE)
df_coloc_all2 <- data.frame(stats = c(t(df_coloc2)), stringsAsFactors=FALSE)
df_total_all2 <- data.frame(stats = c(t(df_tot2)), stringsAsFactors=FALSE)
df_A_coloc_N_hunu_all2 <- data.frame(stats = c(t(df_A_coloc_N_hunu2)), stringsAsFactors=FALSE)
df_N_coloc_A_hunu_all2 <- data.frame(stats = c(t(df_N_coloc_A_hunu2)), stringsAsFactors=FALSE)
df_N_coloc_N_hunu_all2 <- data.frame(stats = c(t(df_N_coloc_N_hunu2)), stringsAsFactors=FALSE)
df_A_col1a1_A_hunu_all2 <- data.frame(stats = c(t(df_A_col1a1_A_hunu2)), stringsAsFactors=FALSE)

df_A_col1a1_A_hunu_all2


#g1 and g2 used when taking means of each column (animal) whereas v1 and v2 used when extracting the column values

v1 <-  dplyr::pull(df_N_coloc_N_hunu_all, stats)
v2 <- dplyr::pull(df_N_coloc_N_hunu_all2, stats)

# N(colocalised)/N(hunu) gained via own normalisation method
norm_N_COL1A1_HUNU_N_HUNU_1 = (((df1$N_COL1A1_HUNU-(df1$N_HUNU/2))/df1$N_HUNU))
norm_N_COL1A1_HUNU_N_HUNU_2 = (((df2$N_COL1A1_HUNU-(df2$N_HUNU/2))/df2$N_HUNU))
#need to put things into a dataframe since the normalised variables are vectors atm
mean_anim_id <-  c(rep('A7',3), rep('A2',4), rep('A20',4), rep('A14',3))
mean_anim_id2 <- c(rep('A25',4), rep('A18', 4), rep('A6',5), rep('A12', 5))
values <- c(norm_N_COL1A1_HUNU_N_HUNU_1, norm_N_COL1A1_HUNU_N_HUNU_2)

animals <- c(rep('A7',3), rep('A2',4), rep('A20',4), rep('A14',3))

animals2 <- c(rep('A25',4), rep('A18', 4), rep('A6',5), rep('A12', 5))

normalised_n_coloc_n_hunu1 <- data.frame(animals, norm_N_COL1A1_HUNU_N_HUNU_1)
normalised_n_coloc_n_hunu2 <- data.frame(animals2, norm_N_COL1A1_HUNU_N_HUNU_2)

v <- normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1
A7 <- c(head(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1,3), 0)
A2 <- c(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1[4:7])
A20 <- c(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1[8:11])
A14 <- c(normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1[11:14])

df_norm1 <- data.frame(A7,A2,A20,A14)

A25 <- c(head(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2,4), 0)
A18 <- c(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2[5:8], 0)
A6 <- c(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2[9:13])
A12 <- c(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2[14:18])
df_norm2 <- data.frame(A25,A18,A6,A12)

g1_norm <- colMeans(df_norm1)
g2_norm <- colMeans(df_norm2)

#to create a data frame with the normalised values when mean of each animal are calculated
values_mean_norm <- c(g1_norm, g2_norm)
groups_mean <- c(rep('1',4), rep('2',4))
df_norm_mean <- data.frame(groups_mean,values_mean_norm)

library(ggplot2)
#########

p <- ggplot(df_norm_mean, aes(x=df_means$groups_mean, y=df_means$values_mean, fill=groups_mean)) + 
  geom_violin(alpha = 0.5) +
  geom_boxplot(width=0.1, fill="white")+
  geom_point(position = position_jitter(seed = 1, width = 0.2))+
  labs(x="Group", y = "Normalised N(COL1A1+HuNu+)/N(HuNu+)")

p
p <- p + theme_bw()

p
####


g1 <- colMeans(normalised_n_coloc_n_hunu1)
g2<- colMeans(normalised_n_coloc_n_hunu2)


v1 <-  dplyr::pull(norm_N_COL1A1_HUNU_N_HUNU_1, stats)
v2 <- dplyr::pull(norm_N_COL1A1_HUNU_N_HUNU_2, stats)

v1 <- norm_N_COL1A1_HUNU_N_HUNU_1
v2 <- norm_N_COL1A1_HUNU_N_HUNU_2


g1 <- colMeans(df_A_col1a1_A_hunu)
g2<- colMeans(df_A_col1a1_A_hunu2)

a = normalised_n_coloc_n_hunu1$norm_N_COL1A1_HUNU_N_HUNU_1
g1 <- colMeans(a)
g2<- colMeans(normalised_n_coloc_n_hunu2$norm_N_COL1A1_HUNU_N_HUNU_2)

#g1 <- t(g1)
#g2 <- t(g2)
#v1_m <-  dplyr::pull(g1)
#v2_m <- dplyr::pull(g1)

values_mean <- c(g1,g2)
groups_mean <- c(rep('1',4), rep('2',4))

df_means <- data.frame(groups_mean,values_mean)
df_means
library(ggplot2)
#########

p <- ggplot(df_means, aes(x=df_means$groups_mean, y=df_means$values_mean, fill=groups_mean)) + 
  geom_violin(alpha = 0.5) +
  geom_boxplot(width=0.1, fill="white")+
  geom_point(position = position_jitter(seed = 1, width = 0.2))+
  labs(x="Group", y = "A(COL1A1)/A(HuNu)")

p
p <- p + theme_bw()

p
########


values <- c(v1,v2)
values <- c(norm_N_COL1A1_HUNU_N_HUNU_1, norm_N_COL1A1_HUNU_N_HUNU_2)
values
group <- c(rep('1',14), rep('2',18))
df <- data.frame(group, values)

library("tidyverse")
library(tidyr)


library(ggplot2)
#########

p <- ggplot(df, aes(x=df$group, y=df$values, fill=group)) + 
  geom_violin(alpha = 0.5) +
  geom_boxplot(width=0.1, fill="white")+
  geom_point(position = position_jitter(seed = 1, width = 0.2))+
  labs(x="Group", y = "normalised N(COL1A1+HuNu+)/N(HuNu)")

  
p <- p + theme_bw()

p

norm_N_COL1A1_HUNU_N_HUNU_1 = (((df1$N_COL1A1_HUNU-(df1$N_HUNU/2))/df1$N_HUNU))
norm_N_COL1A1_HUNU_N_HUNU_2 = (((df2$N_COL1A1_HUNU-(df2$N_HUNU/2))/df2$N_HUNU))

d <- cbind(norm_N_COL1A1_HUNU_N_HUNU_1,norm_N_COL1A1_HUNU_N_HUNU_2)

values <- c(df_A_col1a1_A_hunu_all, df_A_col1a1_A_hunu_all2)
group <- c(rep('1',16), rep('2',20))
df <- data.frame(group, values)

#########

p <- ggplot(df_means, aes(x=df_means$groups_mean, y=df_means$values_mean, fill=groups_mean)) + 
  geom_violin(alpha = 0.5) +
  geom_boxplot(width=0.1, fill="white")+
  geom_point(position = position_jitter(seed = 1, width = 0.2))+
  labs(x="Group", y = "A(COL1A1)/A(HuNu)")

p
p <- p + theme_bw()

p
