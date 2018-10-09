using System;
using System.IO;
using System.Reflection;
using System.Threading;

using CmdParser;
using Microsoft.Search.UHRS.Client;
using Microsoft.Search.UHRS.Client.ManagementReference;
using Microsoft.Search.UHRS.Client.ManagementExReference;
using Microsoft.Search.UHRS.Client.ServiceReference;
using Microsoft.Search.UHRS.Client.AuditReference;
using Microsoft.Search.UHRS.Client.DataStreamingReference;
using Microsoft.Search.UHRS.Client.AuthReference;
using System.Diagnostics;

namespace CVUHRS
{
    static class Program
    {
        public static readonly int[] AllHitappIds = { 34524, 34872, 34879, 35716, 35851, 35852, 35853 };

        public static ManagementClient Management
        {
            get
            {
                return UHRSServiceClientSetup.GetManagement();
            }
        }

        public static DataStreamingClient Streaming
        {
            get
            {
                return UHRSServiceClientSetup.GetStreaming();
            }
        }

        public static void DumpProperties(object obj)
        {
            PropertyInfo[] props = obj.GetType().GetProperties();
            Console.WriteLine(props.Length);
            foreach (PropertyInfo pi in props)
                Console.WriteLine(pi.PropertyType + " " + pi.Name + " : " + pi.GetValue(obj, null));
            Console.WriteLine();
        }

        public static void AuthenticateLegacy(string UserName, string Password)
        {
            string serverAddress = @"prod.uhrs.playmsn.com";

            UHRSServiceClientSetup.SetUserName(UserName); //necessary because the second call doesn't actually use its parameter...
            UHRSServiceClientSetup.SetPassword(Password); //necessary because the second call doesn't actually use its parameter...
            UHRSServiceClientSetup.SetServerAddress(serverAddress); //necessary because the second call doesn't actually use its parameter...
            try
            {
                UHRSServiceClientSetup.LogInLegacy();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("An error occurred when trying to log in using current credentials!");
                Console.Error.WriteLine(ex);
            }
        }

        private static void Error(string Message)
        {
            Console.Error.WriteLine(Message);
            UHRSServiceClientSetup.Reset();
            Environment.Exit(-1);
        }

        public static void Authenticate()
        {
            string serverAddress = @"prod.uhrs.playmsn.com"; //verify that stuff works with INT
            try
            {
                UHRSServiceClientSetup.SetServerAddress(serverAddress); //necessary because the second call doesn't actually use its parameter...
                UHRSServiceClientSetup.WinLogin(serverAddress);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("An error occurred when trying to log in using current credentials!");
                Console.Error.WriteLine(ex);
            }
        }

        public static bool TryDownloadTaskWrapped(int taskGroupId, int taskId, string resultPath)
        {
            const int retryIntervalSec = 10;
            const int numMaxTries = 50;
            Console.WriteLine("[{0}] Downloading task {1}, {2}", DateTime.Now, taskGroupId, taskId);
            string sf = Streaming.PreparePlainTaskFile(taskGroupId, taskId);
            var count = 0;
            bool isSuccess = false;
            while (!isSuccess && count++ < numMaxTries)
            {
                isSuccess = Streaming.IsFileReady(sf, taskGroupId);
                Console.WriteLine($"Number of tries: {count}, Success: {isSuccess}");
                if (!isSuccess)
                {
                    Thread.Sleep(1000 * retryIntervalSec);
                }
            }
            if (isSuccess)
            {
                using (var s = Streaming.DownloadTaskFile(sf, taskGroupId))
                {
                    using (Stream file = new FileStream(resultPath, FileMode.Create, FileAccess.Write))
                    {
                        s.CopyTo(file);
                        file.Close();
                    }
                }
                return true;
            }
            Console.WriteLine($"Sync timed out after {retryIntervalSec * numMaxTries / 60} min");
            return false;
        }

        public static bool TryUploadTaskWrapped(string taskFilePath, double consensusThreshold,
            int numJudgements, int taskGroupId, string taskName, out int taskId)
        {
            int minConsensus = 0;
            int maxConsensus = 0;
            if (consensusThreshold > 0)
            {
                maxConsensus = numJudgements;
                minConsensus = (int)Math.Ceiling(numJudgements * consensusThreshold);
            }
            try
            {
                using (Stream reader = new FileStream(
                UHRSServiceClientSetup.CheckFileForIllegalCharacters(taskFilePath), FileMode.Open))
                {
                    taskId = Streaming.SubmitPlainTask(false, true, false, false, -1, 1.0, numJudgements, null, 1000,
                        false, taskGroupId, taskName, true, minConsensus, maxConsensus,
                        PlainTaskUploadType.Normal, null, false, 0.0, reader);
                    Console.WriteLine($"Uploaded task {taskId}: {taskName}");
                }
                return true;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Exception: {e.Message}");
                taskId = -1;
            }
            return false;
        }

        public class ArgsUploadFromFolder
        {
            [Argument(ArgumentType.Required, HelpText = "Task group Id")]
            public int taskGroupId = -1;
            [Argument(ArgumentType.Required, HelpText = "Folder of task files to be uploaded")]
            public string folderPath = null;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Prefix of filenames that should be uploaded (default: empty string)")]
            public string filePrefix = "";
            [Argument(ArgumentType.Required, HelpText = "TSV file to write uploaded task id and names")]
            public string taskIdNameFile = null;
            [Argument(ArgumentType.Required, HelpText = "Number of judgments required per HIT")]
            public int numJudgment = 0;
            [Argument(ArgumentType.AtMostOnce, HelpText = "Specify if want to use consensus mode (default: 0.0)")]
            public double consensusThreshold = 0.0;
        }

        public static void UploadTasksFromFolder(ArgsUploadFromFolder cmd)
        {
            string[] filePaths = Directory.GetFiles(cmd.folderPath, $"{cmd.filePrefix}*", SearchOption.TopDirectoryOnly);
            Console.WriteLine(filePaths);
            using (var writer = new StreamWriter(cmd.taskIdNameFile))
            {
                string taskName;
                int taskId = 0;
                foreach (string filePath in filePaths)
                {
                    taskName = Path.GetFileNameWithoutExtension(filePath);
                    if (!TryUploadTaskWrapped(filePath, cmd.consensusThreshold, cmd.numJudgment, cmd.taskGroupId,
                        taskName, out taskId))
                    {
                        throw new Exception($"Failed to upload {cmd.taskGroupId}: {cmd.folderPath}");
                    }
                    writer.WriteLine(string.Format("{0}\t{1}", taskId, taskName));
                }
            }
        }

        public class ArgsDownloadToFolder
        {
            [Argument(ArgumentType.Required, HelpText = "Task group Id")]
            public int taskGroupId = -1;
            [Argument(ArgumentType.Required, HelpText = "Output folder for downloaded files")]
            public string folderPath = null;
            [Argument(ArgumentType.Required, HelpText = "TXT file of task ids to be downloaded, the first column of each line should be id")]
            public string taskIdFile = null;
        }

        public static void DownloadTasksToFolder(ArgsDownloadToFolder cmd)
        {
            var lines = File.ReadAllLines(cmd.taskIdFile);
            int taskId;
            foreach (string line in lines)
            {
                var parts = line.Split('\t');
                if (!int.TryParse(parts[0], out taskId))
                {
                    throw new Exception($"Fail to parse task ID {parts[0]} in file {cmd.taskIdFile}");
                }
                string resultPath = Path.Combine(cmd.folderPath, taskId.ToString());
                if (!TryDownloadTaskWrapped(cmd.taskGroupId, taskId, resultPath))
                {
                    // TODO: implement retry logic
                    throw new Exception($"Fail to download {taskId}: {resultPath}");
                }
            }
        }

        public class ArgsBlockSingleJudge
        {
            [Argument(ArgumentType.Required, HelpText = "Judge/Worker id")]
            public int judgeId = -1;
        }

        private static void BlockJudgeHelper(int judgeId)
        {
            foreach (int hitAppId in AllHitappIds)
            {
                Management.SetJudgeHitAppState(judgeId, hitAppId, -1, false);
            }
            Console.WriteLine($"Blocked judge: {judgeId} in all HitApps");
        }

        public static void BlockSingleJudge(ArgsBlockSingleJudge cmd)
        {
            BlockJudgeHelper(cmd.judgeId);
        }

        public class ArgsBlockJudges
        {
            [Argument(ArgumentType.Required, HelpText = "TXT file of judge ids to be blocked")]
            public string filepath = null;
        }

        public static void BlockJudges(ArgsBlockJudges cmd)
        {
            var judgeIds = File.ReadAllLines(cmd.filepath);
            foreach (string judgeIdStr in judgeIds)
            {
                int judgeId;
                if (Int32.TryParse(judgeIdStr, out judgeId))
                {
                    BlockJudgeHelper(judgeId);
                }
                else
                {
                    Console.WriteLine($"cannot recognize JudgeID: {judgeIdStr}");
                }
            }
        }

        public class ArgsTaskState
        {
            [Argument(ArgumentType.Required, HelpText = "Task group id")]
            public int taskGroupId = -1;
            [Argument(ArgumentType.Required, HelpText = "Task id")]
            public int taskId = -1;
        }

        public static void GetTaskState(ArgsTaskState cmd)
        {
            Microsoft.Search.UHRS.Client.ManagementReference.SimpleTask task
                = Management.GetTask(cmd.taskGroupId, cmd.taskId);
            Console.WriteLine($"{task.State} {task.JudgmentsDone} {task.JudgmentsTotal}");
        }

        static int Main(string[] args)
        {
            //use my windows sign in to sign into the UHRS portal
            Authenticate();

            ParserX.AddTask<ArgsUploadFromFolder>(UploadTasksFromFolder, "Upload tasks from folder, write task ids and names");
            ParserX.AddTask<ArgsDownloadToFolder>(DownloadTasksToFolder, "Download tasks to folder");
            ParserX.AddTask<ArgsBlockSingleJudge>(BlockSingleJudge, "Block a single judge/worker on all HitApp");
            ParserX.AddTask<ArgsBlockJudges>(BlockJudges, "Block judges/workers on all HitApp");
            ParserX.AddTask<ArgsTaskState>(GetTaskState, "Print task state, #judgments done, #judgments required");

            if (ParserX.ParseArgumentsWithUsage(args))
            {
                Stopwatch timer = Stopwatch.StartNew();
                ParserX.RunTask();
                timer.Stop();
                Console.WriteLine("Time used: {0}", timer.Elapsed);
            }
            return 0;
        }
    }

}
