/*
    This program identifies all the anagrams in a file 
    and writes a file where the grouped anagrams are sorted
    in the descending order of the number of anagrams for a 
    particular group of letters.
*/

import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;


public class AnagramSorter {
    public static class WordMapper extends Mapper<LongWritable, Text, Text, Text> {

        private Text word = new Text();
        private Text sorted_word = new Text();
        public void map(LongWritable key, Text value, Context context) throws IOException {

            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line);
            while(tokenizer.hasMoreTokens()) {
                String token = tokenizer.nextToken();
                char[] letters = token.toCharArray();
                Arrays.sort(letters);
                sorted_wrd.set(new String(letters));
                word.set(token);
                context.write(sorted_wrd, word);
            }

        }
    }

    public static class AnagramReducer extends Reducer<Text, Text, NullWritable, Text> {

        private Text word = new Text();
        public void reduce(Text key, Iterator<Text> values, Context context) throws IOException {

            String val = "";
            while(values.hasNext()) {
                value = values.next();
                val += value + "\t";
            }
            val.trim();
            word.set(val);
            context.write(NullWritable.get(), word);

        }

    }

    public static class AnagramCountMapper extends Mapper<LongWritable, Text, LongWritable, Text> {

        public void map(LongWritable key, Text value, Context context) throws IOException {

            String line = value.toString();
            line = line.trim();
            String[] values = line.split("\t");
            word.set(value);
            context.collect(new LongWritable(-1 * values.length), word);
        }
    }

    public static class AnagramCountReducer extends Reducer<LongWritable, Text, NullWritable, Text> {

        private Text result = new Text();
        public void reduce(Text key, Iterator<Text> values, Context context) throws IOException {
            for (Text value : values) {
                result.set(value);
                context.write(NullWritable.get(), result);
            }
            
        }

    }

    public static void main(String[] args) throws Exception {
        JobControl jobControl = new JobControl("Anagram");
        Configuration conf1 = new Configuration();

        Job job1 = Job.getInstance(conf1);
        job1.setJobName("Anagram Identifier");
        job1.setJarByClass(AnagramSorter.class);
        job1.setMapperClass(WordMapper.class);
        job1.setReducerClass(AnagramReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.addOutputPath(job1, new Path(args[1] + "/temp"));

        ControlledJob controlledJob1 = new ControlledJob(conf1);
        controlledJob1.setJob(job1);

        jobControl.addJob(controlledJob1);

        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2);
        job2.setJobName("Anagram Sorter");
        job2.setJarByClass(AnagramSorter.class);
        job2.setMapperClass(AnagramCountMapper.class);
        job2.setReducerClass(AnagramCountReducer.class);
        job2.setNumReduceTasks(1);
        job2.setOutputKeyClass(LongWritable.class);
        job2.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job2, new Path(args[1] + "/temp"));
        FileOutputFormat.addOutputPath(job2, new Path(args[1] + "/final"));

        ControlledJob controlledJob2 = new ControlledJob(conf2);
        controlledJob2.setJob(job2);

        controlledJob2.addDependingJob(controlledJob1);
        jobControl.addJob(controlledJob2);

        Thread jobControlThread = new Thread(jobControl);
        jobControlThread.start();
        while(!jobControl.allFinished())
        {
            Thread.sleep(5000);
        }

        System.exit(0);
    }
}