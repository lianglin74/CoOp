using System;
using System.IO;
using System.Reflection;
using System.Threading;

using Microsoft.Search.UHRS.Client;
using Microsoft.Search.UHRS.Client.ManagementReference;
using Microsoft.Search.UHRS.Client.ManagementExReference;
using Microsoft.Search.UHRS.Client.ServiceReference;
using Microsoft.Search.UHRS.Client.AuditReference;
using Microsoft.Search.UHRS.Client.DataStreamingReference;
using Microsoft.Search.UHRS.Client.AuthReference;

namespace CVUHRS
{
    static class Program {
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
                    using (Stream file = new FileStream(resultPath, FileMode.CreateNew, FileAccess.Write))
                    {
                        s.CopyTo(file);
                        file.Close();
                    }
                }
                return true;
            }
            Console.WriteLine($"Sync timed out after {retryIntervalSec*numMaxTries/60} min");
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

        public static void UploadTasksFromFolder(int taskGroupId, string folderPath, string filePrefix,
            string taskIdNameFile, double consensusThreshold, int numJudgement)
        {
            string[] filePaths = Directory.GetFiles(folderPath, $"{filePrefix}*", SearchOption.TopDirectoryOnly);
            Console.WriteLine(filePaths);
            using (var writer = new StreamWriter(taskIdNameFile))
            {
                string taskName;
                int taskId = 0;
                foreach (string filePath in filePaths)
                {
                    taskName = Path.GetFileNameWithoutExtension(filePath);
                    if (!TryUploadTaskWrapped(filePath, consensusThreshold, numJudgement, taskGroupId,
                        taskName, out taskId))
                    {
                        throw new Exception($"Failed to upload {taskGroupId}: {folderPath}");
                    }
                    writer.WriteLine(string.Format("{0}\t{1}", taskId, taskName));
                }
            }
        }

        public static void DownloadTasksToFolder(int taskGroupId, string folderPath, string taskIdNameFile)
        {
            var lines = File.ReadAllLines(taskIdNameFile);
            int taskId;
            foreach (string line in lines)
            {
                var parts = line.Split('\t');
                if (!int.TryParse(parts[0], out taskId))
                {
                    throw new Exception($"Fail to parse task ID {parts[0]} in file {taskIdNameFile}");
                }
                string resultPath = Path.Combine(folderPath, taskId.ToString());
                if (!TryDownloadTaskWrapped(taskGroupId, taskId, resultPath))
                {
                    // TODO: implement retry logic
                    throw new Exception($"Fail to download {taskId}: {resultPath}");
                }
            }
        }

        public static void BlockSingleJudge(int judgeId)
        {
            foreach (int hitAppId in AllHitappIds)
            {
                Management.SetJudgeHitAppState(judgeId, hitAppId, -1, false);
            }
            Console.WriteLine($"Blocked judge: {judgeId} in all HitApps");
        }

        public static void BlockJudges(string filePath)
        {
            var judgeIds = File.ReadAllLines(filePath);
            foreach (string judgeIdStr in judgeIds)
            {
                int judgeId;
                if (Int32.TryParse(judgeIdStr, out judgeId))
                {
                    BlockSingleJudge(judgeId);    
                }
                else
                {
                    Console.WriteLine($"cannot recognize JudgeID: {judgeIdStr}");
                }
            }
        }

        public static void GetTaskState(int taskGroupId, int taskId)
        {
            Microsoft.Search.UHRS.Client.ManagementReference.SimpleTask task
                = Management.GetTask(taskGroupId, taskId);
            Console.WriteLine($"{task.State} {task.JudgmentsDone} {task.JudgmentsTotal}");
        }

        static int Main(string[] args)
        {
            if (args == null)
            {
                throw new ArgumentException("Empty arguments");
            }
            //use my windows sign in to sign into the UHRS portal
            Authenticate();
            string taskType = args[0].ToLower();
            
            if (taskType == "upload_from_folder")
            {
                // args: taskGroupId(int), pathToTaskFolder(str), filePrefix(str),
                // pathToTaskIdNameMap(str), consensusThreshold(double), numJudgement(int)
                int taskGroupId, numJudgement;
                double consensusThreshold;
                if (!int.TryParse(args[1], out taskGroupId))
                {
                    throw new ArgumentException($"cannot parse task group id {args[0]}");
                }
                if (!double.TryParse(args[5], out consensusThreshold))
                {
                    throw new ArgumentException($"cannot parse consensus threshold {args[4]}");
                }
                if (!int.TryParse(args[6], out numJudgement))
                {
                    throw new ArgumentException($"cannot parse number of judgement {args[5]}");
                }
                string pathToTaskFolder = args[2];
                string filePrefix = args[3];
                string taskIdNameMap = args[4];
                UploadTasksFromFolder(taskGroupId, pathToTaskFolder, filePrefix, 
                    taskIdNameMap, consensusThreshold, numJudgement);
                return 0;
            }

            if (taskType == "download_to_folder")
            {
                // args: taskGroupId(int), pathToTaskFolder(str), taskIdNameMap(str)
                int taskGroupId;
                string pathToTaskFolder, taskIdNameMap;
                if (!int.TryParse(args[1], out taskGroupId))
                {
                    throw new ArgumentException($"cannot parse task group id {args[0]}");
                }
                pathToTaskFolder = args[2];
                taskIdNameMap = args[3];
                DownloadTasksToFolder(taskGroupId, pathToTaskFolder, taskIdNameMap);
                return 0;
            }

            if (taskType == "block_judges")
            {
                string filepath = args[1];
                BlockJudges(filepath);
                return 0;
            }

            if (taskType == "block_judge")
            {
                int judgeId;
                if (!int.TryParse(args[1], out judgeId))
                {
                    throw new ArgumentException($"cannot parse judge id {args[1]}");
                }
                BlockSingleJudge(judgeId);
                return 0;
            }

            if (taskType == "get_task_state")
            {
                int taskGroupId, taskId;
                if (!int.TryParse(args[1], out taskGroupId))
                {
                    throw new ArgumentException($"cannot parse task group id {args[1]}");
                }
                if (!int.TryParse(args[2], out taskId))
                {
                    throw new ArgumentException($"cannot parse task id {args[2]}");
                }
                GetTaskState(taskGroupId, taskId);
                return 0;
            }

            throw new NotImplementedException($"Unrecognized task: {taskType}");
        }
    }
      
}
