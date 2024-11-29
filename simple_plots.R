rm(list = ls())
cat("\014")
setwd('~/Documents/Results')

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)

NAMES <- c('Evolved','Deterministic','Equal')
TASKS <- c('heart_disease', 'student_math', 'us_crime', 'nlsy', 'compas', 'law_school','pmad_phq', 'pmad_epds')
SHAPE <- c(21,24,22)
cb_palette <- c('#D81B60','#1E88E5','#FFC107')
TSIZE <- 19
data_dir <- ''

p_theme <- theme(
  plot.title = element_text( face = "bold", size = 22, hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  legend.title=element_text(size=22),
  legend.text=element_text(size=23),
  axis.title = element_text(size=23),
  axis.text = element_text(size=19),
  legend.position="bottom",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  linewidth = 0.5, linetype = "solid")
)

testing <- read.csv('hv_values11/hv_test.csv', header = TRUE, stringsAsFactors = FALSE)
testing$exp <- gsub('Evolved Weights', 'Evolved', testing$ex)
testing$exp <- gsub('Deterministic Weights', 'Deterministic', testing$ex)
testing$exp <- gsub('Equal Weights', 'Equal', testing$ex)
testing$exp <- gsub('Evolved_Weights_Holdout', 'HoldoutLexicase', testing$ex)
testing$exp <- gsub('Evolved_Weights_Lexidate', 'FairLexidate', testing$ex)
testing$exp <- factor(testing$exp, levels = NAMES)


# testing 

# task 1

task_1_p <- filter(testing, dataset == TASKS[1]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[1])+
  p_theme + coord_flip()

# task 2

task_2_p <- filter(testing, dataset == TASKS[2]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[2])+
  p_theme+ coord_flip()

# task 3

task_3_p <- filter(testing, dataset == TASKS[3]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[3])+
  p_theme+ coord_flip()

# task 4

task_4_p <- filter(testing, dataset == TASKS[4]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[4])+
  p_theme+ coord_flip()

# task 5

task_5_p <- filter(testing, dataset == TASKS[5]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[5])+
  p_theme+ coord_flip()

# task 6

task_6_p <- filter(testing, dataset == TASKS[6]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[6])+
  p_theme+ coord_flip()

# task 7

task_7_p <- filter(testing, dataset == TASKS[7]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[7])+
  p_theme+ coord_flip()

# task 8

task_8_p <- filter(testing, dataset == TASKS[8]) %>%
  ggplot(., aes(x = exp, y = hv, color = exp, fill = exp, shape = exp)) +
  geom_flat_violin(position = position_nudge(x = .1, y = 0), scale = 'width', alpha = 0.2, width = 1.5) +
  geom_boxplot(color = 'black', width = .07, outlier.shape = NA, alpha = 0.0, size = 1.0, position = position_nudge(x = .16, y = 0)) +
  geom_point(position = position_jitter(width = 0.02, height = 0.0001), size = 1.5, alpha = 1.0) +
  scale_y_continuous(
    name="Volume",
  ) +
  scale_x_discrete(
    name="Population Size"
  )+
  scale_shape_manual(values=SHAPE)+
  scale_colour_manual(values = cb_palette, ) +
  scale_fill_manual(values = cb_palette) +
  ggtitle(TASKS[8])+
  p_theme+ coord_flip()


# legend
legend <- cowplot::get_legend(
  task_1_p +
    guides(
      shape=guide_legend(ncol=1,title="Weight Strategy",reverse = TRUE,title.position="left", title.hjust = 0.5),
      color=guide_legend(ncol=1,title="Weight Strategy",reverse = TRUE,title.position="left", title.hjust = 0.5),
      fill=guide_legend(ncol=1,title="Weight Strategy",reverse = TRUE,title.position="left", title.hjust = 0.5)
    ) +
    theme(
      legend.position = "top",
      legend.box="verticle",
      legend.justification="center"
    )
)


col1_theme <- theme(legend.position = "none", axis.title.y = element_blank(), axis.title.x = element_blank(),
                    axis.text.y = element_blank(), axis.ticks.y = element_blank())

col1 <- plot_grid(
  task_1_p + ggtitle(TASKS[1]) + col1_theme,
  task_2_p + ggtitle(TASKS[2]) + col1_theme,
  task_3_p + ggtitle(TASKS[3]) + col1_theme,
  task_4_p + ggtitle(TASKS[4]) + col1_theme,
  ncol=4,
  rel_heights = c(1.0,1.0,1.0,1.0),
  label_size = TSIZE
)

col2 <- plot_grid(
  task_5_p + ggtitle(TASKS[5]) + col1_theme,
  task_6_p + ggtitle(TASKS[6]) + col1_theme,
  task_7_p + ggtitle(TASKS[7]) + col1_theme,
  task_8_p + ggtitle(TASKS[8]) + col1_theme,
  ncol=4,
  rel_heights = c(1.0,1.0,1.0,1.0),
  label_size = TSIZE
)

colb <-  theme(legend.position = "none", axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())


fig <- plot_grid(
  ggdraw() + draw_label("Hypervolume for Pareto Front on Test Set", fontface='bold', size = 24) + p_theme,
  col1,
  col2,
  #col3,
  legend,
  nrow=5,
  rel_heights = c(0.15,1.0,1.0,1.0,0.3),
  label_size = TSIZE
)

fig

save_plot(
  paste(filename ="result-h11_simple.pdf"),
  fig,
  base_width=20,
  base_height=11
)

