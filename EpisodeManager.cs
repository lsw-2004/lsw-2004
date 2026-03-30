using UnityEngine;
using UnityEngine.AI;

/// <summary>
/// 行人路径配置：在指定的世界坐标区域内，生成一条直线路径（P1、P2、P3...）
/// </summary>
[System.Serializable]
public class PedestrianPath
{
    public string name = "Pedestrian 1";

    [Tooltip("这个行人的根节点（有 NavMeshAgent / 巡航脚本的那个），一般就是行人模型的最外层 Transform")]
    public Transform pedestrianRoot;

    [Tooltip("这个行人使用的路径点（P1、P2、P3...），在 Inspector 里按顺序拖进来")]
    public Transform[] waypoints;

    [Header("路径随机区域（世界坐标）")]
    public Vector3 areaCenterWorld;
    public Vector3 areaSize = new Vector3(4f, 0f, 2f);
}

/// <summary>
/// Episode 管理：
/// - 随机 Robot 初始位置（严格在 NavMesh 上）
/// - 随机 Goal 位置（严格在 NavMesh 上）
/// - 随机每个行人的路径点
/// - Reset 由外部 Python 或调试按键 R 触发
/// </summary>
public class EpisodeManager : MonoBehaviour
{
    [Header("机器人 & 控制")]
    public Transform robot;
    public Rigidbody robotRb;
    public RobotController robotController;

    [Header("目标点")]
    public Transform goal;

    [Header("随机区域 - 机器人（世界坐标）")]
    public Vector3 robotAreaCenterWorld = Vector3.zero;
    public Vector3 robotAreaSize = new Vector3(2f, 0f, 8f);

    [Header("随机区域 - 目标点（世界坐标）")]
    public Vector3 goalAreaCenterWorld = new Vector3(0f, 0f, 8f);
    public Vector3 goalAreaSize = new Vector3(2f, 0f, 8f);

    [Header("Episode 设置")]
    public float minRobotGoalDistance = 5f;
    public float maxEpisodeTime = 30f;
    public float goalRadius = 0.5f;

    [Header("NavMesh 采样设置")]
    [Tooltip("随机点向最近 NavMesh 投影时允许的最大搜索距离")]
    public float navMeshSampleDistance = 2.0f;

    [Tooltip("robot / goal 离边界或墙体至少保留的余量（近似通过 NavMesh.Raycast 检查）")]
    public float navMeshWallClearance = 0.35f;

    [Tooltip("随机采样最多尝试次数")]
    public int maxSpawnTries = 50;

    [Header("行人路径随机化")]
    public PedestrianPath[] pedestrians;

    private float _robotFixedY;
    private float _goalFixedY;
    private Quaternion _robotStartRot;
    private Vector3 _robotStartPos;
    private Vector3 _goalStartPos;

    void Start()
    {
        if (robot != null)
        {
            _robotFixedY = robot.position.y;
            _robotStartRot = robot.rotation;
            _robotStartPos = robot.position;
        }

        if (goal != null)
        {
            _goalFixedY = goal.position.y;
            _goalStartPos = goal.position;
        }

        ResetEpisode();
    }

    void Update()
    {
        // 手动调试 reset
        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetEpisode();
        }
    }

    /// <summary>
    /// 外部调用：一局重置
    /// </summary>
    public void ResetEpisode()
    {
        // 0) 停车 & 清零
        if (robotController != null)
            robotController.Stop();

        if (robotRb != null)
        {
            robotRb.velocity = Vector3.zero;
            robotRb.angularVelocity = Vector3.zero;
        }

        // 1) 随机 Goal（严格在 NavMesh 上）
        Vector3 newGoalPos = (goal != null)
            ? SampleStrictNavMeshPosition(goalAreaCenterWorld, goalAreaSize, _goalFixedY, _goalStartPos)
            : Vector3.zero;

        if (goal != null)
        {
            goal.position = newGoalPos;
        }

        // 2) 随机 Robot（严格在 NavMesh 上，并且与 Goal 保持最小距离）
        if (robot != null)
        {
            Vector3 newRobotPos = SampleRobotSpawnWithGoalConstraint(
                robotAreaCenterWorld,
                robotAreaSize,
                _robotFixedY,
                _robotStartPos,
                (goal != null) ? goal.position : _robotStartPos
            );

            if (robotController != null)
                robotController.Stop();

            if (robotRb != null)
            {
                robotRb.velocity = Vector3.zero;
                robotRb.angularVelocity = Vector3.zero;

                robotRb.position = newRobotPos;
                robotRb.rotation = _robotStartRot;

                robotRb.velocity = Vector3.zero;
                robotRb.angularVelocity = Vector3.zero;
                robotRb.Sleep();
            }
            else
            {
                robot.position = newRobotPos;
                robot.rotation = _robotStartRot;
            }

            Physics.SyncTransforms();
        }

        // 3) 随机行人路径
        RandomizePedestrians();
    }

    /// <summary>
    /// 严格在 NavMesh 上采样。
    /// 失败时不会返回随机 candidate，而是回退到安全点。
    /// </summary>
    private Vector3 SampleStrictNavMeshPosition(
        Vector3 centerWorld,
        Vector3 size,
        float fixedY,
        Vector3 fallbackWorld)
    {
        // 多次随机 + NavMesh 投影
        for (int k = 0; k < maxSpawnTries; k++)
        {
            float rx = Random.Range(-size.x * 0.5f, size.x * 0.5f);
            float rz = Random.Range(-size.z * 0.5f, size.z * 0.5f);
            Vector3 candidate = new Vector3(centerWorld.x + rx, fixedY, centerWorld.z + rz);

            if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleDistance, NavMesh.AllAreas))
            {
                Vector3 p = hit.position;
                p.y = fixedY;

                if (IsPointSafeOnNavMesh(p))
                    return p;
            }
        }

        // 第二层兜底：在区域中心附近找一个可行走点
        if (NavMesh.SamplePosition(centerWorld, out NavMeshHit centerHit, navMeshSampleDistance * 2f, NavMesh.AllAreas))
        {
            Vector3 p = centerHit.position;
            p.y = fixedY;

            if (IsPointSafeOnNavMesh(p))
                return p;
        }

        // 第三层兜底：用初始点/默认点再贴一次 NavMesh
        if (NavMesh.SamplePosition(fallbackWorld, out NavMeshHit fallbackHit, navMeshSampleDistance * 2f, NavMesh.AllAreas))
        {
            Vector3 p = fallbackHit.position;
            p.y = fixedY;
            return p;
        }

        // 最后兜底：返回原始 fallback，但至少不是随机非法点
        return new Vector3(fallbackWorld.x, fixedY, fallbackWorld.z);
    }

    /// <summary>
    /// 采样 robot 出生点，并确保与 goal 至少相距 minRobotGoalDistance
    /// </summary>
    private Vector3 SampleRobotSpawnWithGoalConstraint(
        Vector3 centerWorld,
        Vector3 size,
        float fixedY,
        Vector3 fallbackWorld,
        Vector3 goalPos)
    {
        for (int k = 0; k < maxSpawnTries; k++)
        {
            Vector3 p = SampleStrictNavMeshPosition(centerWorld, size, fixedY, fallbackWorld);
            if (Vector3.Distance(p, goalPos) >= minRobotGoalDistance)
                return p;
        }

        // 如果一直找不到，就返回一个严格合法的 NavMesh 点
        return SampleStrictNavMeshPosition(centerWorld, size, fixedY, fallbackWorld);
    }

    /// <summary>
    /// 近似检查：点是否不太贴近 NavMesh 边界/墙
    /// 用几个方向做短程 NavMesh.Raycast，尽量避免出生在边缘
    /// </summary>
    private bool IsPointSafeOnNavMesh(Vector3 p)
    {
        float c = navMeshWallClearance;
        if (c <= 1e-4f) return true;

        Vector3[] dirs = new Vector3[]
        {
            Vector3.forward,
            Vector3.back,
            Vector3.left,
            Vector3.right,
            (Vector3.forward + Vector3.left).normalized,
            (Vector3.forward + Vector3.right).normalized,
            (Vector3.back + Vector3.left).normalized,
            (Vector3.back + Vector3.right).normalized,
        };

        foreach (var d in dirs)
        {
            Vector3 target = p + d * c;
            if (NavMesh.Raycast(p, target, out NavMeshHit hit, NavMesh.AllAreas))
            {
                return false;
            }
        }

        return true;
    }

    private void RandomizePedestrians()
    {
        if (pedestrians == null) return;

        foreach (var p in pedestrians)
        {
            if (p == null) continue;
            if (p.waypoints == null || p.waypoints.Length == 0) continue;

            int n = p.waypoints.Length;
            float y = p.waypoints[0].position.y;

            float minX = p.areaCenterWorld.x - p.areaSize.x * 0.5f;
            float maxX = p.areaCenterWorld.x + p.areaSize.x * 0.5f;
            float z = p.areaCenterWorld.z + Random.Range(-p.areaSize.z * 0.5f, p.areaSize.z * 0.5f);

            float startX = Random.Range(minX, maxX);
            float endX = Random.Range(minX, maxX);

            int safety = 0;
            while (Mathf.Abs(endX - startX) < 2f && safety < 50)
            {
                endX = Random.Range(minX, maxX);
                safety++;
            }

            for (int i = 0; i < n; i++)
            {
                float t = (n == 1) ? 0f : (float)i / (n - 1);
                float x = Mathf.Lerp(startX, endX, t);
                Vector3 rawPos = new Vector3(x, y, z);

                // 尽量把 waypoint 也贴到 NavMesh 上，避免 patrol 路径出界
                Vector3 navPos = rawPos;
                if (NavMesh.SamplePosition(rawPos, out NavMeshHit wpHit, navMeshSampleDistance, NavMesh.AllAreas))
                {
                    navPos = wpHit.position;
                    navPos.y = y;
                }

                p.waypoints[i].position = navPos;
            }

            if (p.pedestrianRoot != null)
            {
                NavMeshAgent agent = p.pedestrianRoot.GetComponent<NavMeshAgent>();
                Vector3 spawnPos = p.waypoints[0].position;

                if (agent != null)
                {
                    // 若当前不在 NavMesh 上，尝试先把根节点放到最近 NavMesh 点
                    if (!agent.isOnNavMesh)
                    {
                        if (NavMesh.SamplePosition(spawnPos, out NavMeshHit pedHit, navMeshSampleDistance * 2f, NavMesh.AllAreas))
                        {
                            p.pedestrianRoot.position = pedHit.position;
                        }
                        else
                        {
                            p.pedestrianRoot.position = spawnPos;
                        }
                    }

                    if (agent.isOnNavMesh)
                    {
                        agent.Warp(spawnPos);
                        agent.ResetPath();
                    }
                    else
                    {
                        p.pedestrianRoot.position = spawnPos;
                    }
                }
                else
                {
                    p.pedestrianRoot.position = spawnPos;
                }

                PedestrianPatrol patrol = p.pedestrianRoot.GetComponent<PedestrianPatrol>();
                if (patrol != null)
                {
                    patrol.RestartPatrol();
                }
            }
        }
    }

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f, 1f, 0f, 0.25f);
        Gizmos.DrawWireCube(
            robotAreaCenterWorld,
            new Vector3(robotAreaSize.x, 0.1f, robotAreaSize.z)
        );

        Gizmos.color = new Color(1f, 0.5f, 0f, 0.25f);
        Gizmos.DrawWireCube(
            goalAreaCenterWorld,
            new Vector3(goalAreaSize.x, 0.1f, goalAreaSize.z)
        );

        if (pedestrians != null)
        {
            Gizmos.color = new Color(0f, 0.5f, 1f, 0.25f);
            foreach (var p in pedestrians)
            {
                if (p == null) continue;
                Gizmos.DrawWireCube(
                    p.areaCenterWorld,
                    new Vector3(p.areaSize.x, 0.1f, p.areaSize.z)
                );
            }
        }
    }
}