package com.dl4j.utils;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import java.util.ArrayList;
import java.util.Random;

public class Helper {

    /**
     * Generate random integer between a range.
     * @param min   the minimum integer
     * @param max   the maximum integer
     * @return  the generated seed integer
     * */
    public static int generateSeed(final int min, final int max) {
        Random r = new Random();
        return r.ints(min, (max + 1)).findFirst().getAsInt();
    }

    public static void startUIServer(Model model) {
        //Initialize the user interface backend
        final UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored.
        // Here: store in memory.
        final StatsStorage statsStorage = new InMemoryStatsStorage();

        //Attach the StatsStorage instance to the UI:
        // this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        model.setListeners(new ArrayList<TrainingListener>(){{
            add(new ScoreIterationListener(10));
            add(new StatsListener(statsStorage));
        }});
    }
}
