library(broom)
library(tidyr)
library(dplyr)
#library(pscl)
library(pROC)
#library(plyr)
library(caret)
library(stringr)
library(lme4)
library(lmerTest)
library(lattice)
options(tibble.print_max = Inf)
set.seed(123)
#Ler Base de Dados
notas_raw <-read.csv("~/projects/panel/features_engineering/ALL_SCHOOLS_v2.csv")
notas<- notas_raw%>%filter((IN_TP_ESCOLA== 'Municipal+Estadual') & (CO_ANO == 2019))

uf <- ("~/data/geodata/uf.json") %>%
  st_read()%>%select(UF_05, GEOCODIGO)
st_geometry(uf) <- NULL 

notas$CO_UF<- as.factor(notas$CO_UF)
notas<- left_join(notas, uf, by = c("CO_UF" = "GEOCODIGO"))



### Functions 

clip_tail<- function(df) {
  for (i in colnames(df)) {
    if (is.numeric(df[[i]]) & (length(df[[i]]%>%unique())> 2)){ 
      print(i)
      print(max(df[[i]]))
      print(min(df[[i]]))
      print(sd(df[[i]]))
      df[[i]]<- Winsorize(df[[i]],  probs = c(0.025, 0.975), na.rm = FALSE, type = 7)
      print(max(df[[i]]))
      print(min(df[[i]]))
      print(sd(df[[i]]))
    }
  }
  return(df)
}

drop_col_hight_mode<- function(df){
  
  for (i in colnames(df)){
    prop_mode<- sort(-table(df[[i]]))[1]/nrow(df)*(-1) # get mode value
    if (prop_mode > 0.9){
      df<- df%>%dplyr::select(-i)
      print(i)
    }
  }
  return(df)
}

build_target <- function(df){
  df<- df%>%mutate(
    TARGET = ntile(NU_NOTA_GERAL, 4))
  df$TARGET<- if_else(df$TARGET > 2, 1, 0)
  df<- df%>%dplyr::select(-NU_NOTA_GERAL)
  print(paste0(round(mean(df$TARGET)*100, 1), '%', ' upper quartil'))
  return(df)
}

## Only schools elementary infrastructure schools

df1<- notas%>%filter(IN_AGUA_INEXISTENTE == 0 & IN_ESGOTO_INEXISTENTE ==0 & IN_ENERGIA_INEXISTENTE == 0)
#df1<-notas%>%filter(IN_INFRA_ELEMENTAR==1)

## regularizing some quantitify variables that are binary in some years   
df1$QT_EQUIP_COPIADAORA <- if_else(df1$QT_EQUIP_COPIADORA>0, 1, 0)
df1$QT_EQUIP_DVD <- if_else(df1$QT_EQUIP_DVD>0, 1, 0) 
df1$QT_EQUIP_IMPRESSORA <- if_else(df1$QT_EQUIP_IMPRESSORA>0, 1, 0) 
df1$QT_EQUIP_TV <- if_else(df1$QT_EQUIP_TV>0, 1, 0) 

#df1<-df1%>%filter(CO_ANO==2018)
#df1<- build_target(df1)
#df1<- clip_tail(df1)
#df1<- drop_col_hight_mode(df1)
## Select features 
school_features = c(
  'IN_LABORATORIO_INFORMATICA',
  'IN_LABORATORIO_CIENCIAS',
  'IN_SALA_ATENDIMENTO_ESPECIAL',
  'IN_BIBLIOTECA',
  'IN_SALA_LEITURA',
  'IN_BANHEIRO',
  'IN_BANHEIRO_PNE',
  'QT_SALAS_UTILIZADAS',
  'QT_EQUIP_TV',
  'QT_EQUIP_DVD',
  'QT_EQUIP_COPIADORA',
  'QT_EQUIP_IMPRESSORA',
  'QT_COMP_ALUNO',
  'IN_BANDA_LARGA',
  'QT_FUNCIONARIOS',
  'IN_ALIMENTACAO',
  'IN_COMUM_MEDIO_MEDIO',
  'IN_COMUM_MEDIO_INTEGRADO',
  'IN_COMUM_MEDIO_NORMAL',
  'IN_SALA_PROFESSOR',
  'IN_COZINHA',
  'IN_EQUIP_PARABOLICA',
  'IN_QUADRA_ESPORTES',
  'IN_ATIV_COMPLEMENTAR',
  'QT_MATRICULAS' 
)

teacher_features = c('TITULACAO', 'IN_FORM_DOCENTE','NU_LICENCIADOS', 
                     'NU_CIENCIA_NATUREZA','NU_CIENCIAS_HUMANAS', 'NU_LINGUAGENS_CODIGOS', 'NU_MATEMATICA', 
                     'NU_ESCOLAS', 'DIVERSIDADE')

student_features =c('RENDA_PERCAPITA',  'EDU_PAI', 'EDU_MAE', 'NU_IDADE')


non_actionable_features = c('TP_COR_RACA_0.0', 'TP_COR_RACA_1.0',
                            'TP_COR_RACA_2.0', 'TP_COR_RACA_3.0', 'TP_COR_RACA_4.0',
                            'TP_COR_RACA_5.0',  'TP_SEXO')

controls= c( "UF_05", "NU_NOTA_GERAL")

#trainRowNumbers <- createDataPartition(p=0.25)
#train <- df1[trainRowNumbers,]
#test <- df1[-trainRowNumbers,]

df1<-df1%>%select(student_features, school_features, teacher_features, controls)
#df1_level1<- fastDummies::dummy_cols(df1_level1, select_columns = "TP_COR_RACA", remove_first_dummy = TRUE )%>%
  #select(-TP_COR_RACA)

#df1_level2<-df1%>%select(school_features, teacher_features)
df1<- drop_col_hight_mode(df1)
df1<-clip_tail(df1)

###
#df1_level1<- df1_level1%>%group_by(UF_05)%>%
#  mutate(RENDA_PERCAPITA = RENDA_PERCAPITA - mean(RENDA_PERCAPITA),
#         EDU_PAI = EDU_PAI - mean(EDU_PAI),
#         EDU_MAE = EDU_MAE - mean(EDU_MAE),
#         TP_COR_RACA_1 = TP_COR_RACA_1 - mean(TP_COR_RACA_1),
#         TP_COR_RACA_2 = TP_COR_RACA_2 - mean(TP_COR_RACA_2),
#         TP_COR_RACA_3 = TP_COR_RACA_3 - mean(TP_COR_RACA_3),
#         TP_COR_RACA_4 = TP_COR_RACA_4 - mean(TP_COR_RACA_4),
#         TP_COR_RACA_5 = TP_COR_RACA_5 - mean(TP_COR_RACA_5),
#         NU_IDADE = NU_IDADE - mean(NU_IDADE),
#         TP_SEXO = TP_SEXO - mean(TP_SEXO)
#         )%>%ungroup()


#df1<- cbind(df1_level1, df1_level2)

set.seed(123)
pp = preProcess(df1, method = "range")
df1<- predict(pp, df1)
#df1<-build_target(df1)
features_fixed<-names(df1)[names(df1) != 'UF_05']
formula <- as.formula(paste(paste('NU_NOTA_GERAL', paste(features_fixed, collapse="+"), sep="~"), paste("(1|UF_05)"), sep = "+"))

random_effects_3 <- lmer(formula, data=df1)

#for (i in unique(df1$CO_ANO)){
#  data<- df1%>%filter(CO_ANO == i)%>%select(TP_COR_RACA)
#  print(summary(as.factor(data$TP_COR_RACA)))
#  }
re_2016<-ranef(random_effects_3)
op <- options(digits = 4)
options(op)
## as.data.frame() provides RE's and conditional standard deviations:
str(dd <- as.data.frame(re_2016))
nordeste = c("BA", "SE", "AL", "PE", "PB", "RN", "CE", "MA", "PI")
sudeste = c("SP", "MG", "ES", "RJ")
sul = c("RS", "PR", "SC")
norte = c("AM", "RR", "RO", "AP", "PA", "TO", "AC")
centroeste = c("DF", "MS", "MT", "GO")
ne_n<- dd%>%filter(grp%in%nordeste | grp%in%norte)
remain_states<-dd%>%filter(grp%in%sudeste | grp%in%sul| grp%in%centroeste)



p1<- ggplot(dd, aes(y=grp,x=condval)) +
  geom_point(color = "blue") + facet_wrap(~term,scales="free_x")+ 
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank()
  )+
  geom_errorbarh(aes(xmin=condval -2*condsd,
                     xmax=condval +2*condsd), height=0)+
  theme_publish()+
  xlab("Random effects")+ylab(" States")+
  labs(title = "Plot 2")



p2 <- ggplot(remain_states, aes(y = grp, x = condval)) +
  geom_point(color = "blue") +
  facet_wrap(~term, scales = "free_x") +
  theme(strip.background = element_blank(), strip.text.x = element_blank()) +
  geom_errorbarh(aes(xmin = condval - 2 * condsd, xmax = condval + 2 * condsd), height = 0) +
  theme_publish() +
  xlab("Random effects") + ylab("States") 

library(patchwork)

re_plots<-p1 + p2
re_plots

ggsave(filename = "re_plots.png", plot = re_plots, dpi = 400)


