using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// 改动说明：
/// 1. LiDAR 观测现在默认支持 360 维
/// 2. 不再把空 LiDAR 情况下的补零维度写死为 180，而是自动跟随 lidar.numRays
/// 3. 其他逻辑保持不变
/// </summary>
public class NavAgent : Agent
{
    [Header("绑定组件")]
    public RobotController robot;
    public SimpleLidar2D lidar;
    public Transform goal;
    public EpisodeManager episodeManager;

    [Header("仅环境参数，不再负责训练奖励")]
    public float reachGoalRadius = 0.5f;

    [Header("LiDAR 兜底维度（仅当 lidar 为空时使用）")]
    public int fallbackLidarDim = 360;

    private bool collisionFlag = false;
    private float _maxScenarioSize = 30f;

    public bool CollisionFlag => collisionFlag;

    public override void OnEpisodeBegin()
    {
        collisionFlag = false;

        if (episodeManager != null)
            episodeManager.ResetEpisode();

        if (robot != null)
            robot.Stop();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (lidar != null) lidar.ScanOnce();

        // A. LiDAR
        if (lidar != null && lidar.Distances != null)
        {
            float maxD = Mathf.Max(1e-6f, lidar.maxDistance);
            foreach (float d in lidar.Distances)
                sensor.AddObservation(d / maxD);
        }
        else
        {
            int lidarDim = fallbackLidarDim > 0 ? fallbackLidarDim : 360;
            for (int i = 0; i < lidarDim; i++)
                sensor.AddObservation(1f);
        }

        // B. goal info
        Vector3 toGoal = (goal != null) ? (goal.position - transform.position) : Vector3.zero;
        toGoal.y = 0f;
        float dist = toGoal.magnitude;

        Vector3 dir = dist > 1e-6f ? toGoal.normalized : Vector3.zero;
        sensor.AddObservation(dir.x);
        sensor.AddObservation(dir.z);
        sensor.AddObservation(Mathf.Clamp01(dist / _maxScenarioSize));

        float angle = Vector3.SignedAngle(transform.forward, toGoal, Vector3.up);
        sensor.AddObservation(angle / 180f);

        // C. self velocity
        if (robot != null)
        {
            float normLinear = 0f;
            float normAngular = 0f;

            if (robot.maxLinearSpeed > 1e-6f)
                normLinear = robot.CurrentLinearVelocity / robot.maxLinearSpeed;

            if (robot.maxAngularSpeed > 1e-6f)
                normAngular = robot.CurrentAngularVelocityDeg / robot.maxAngularSpeed;

            sensor.AddObservation(normLinear);
            sensor.AddObservation(normAngular);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // D. collision flag
        sensor.AddObservation(collisionFlag ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float v = actions.ContinuousActions[0];
        float w = actions.ContinuousActions[1];

        if (robot != null)
            robot.SetAction(v, w);
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.collider.CompareTag("Wall") ||
            col.collider.CompareTag("Obstacle") ||
            col.collider.CompareTag("Pedestrian"))
        {
            collisionFlag = true;
        }
    }

    public float GetDistanceToGoal()
    {
        if (goal == null) return 0f;
        return Vector3.Distance(transform.position, goal.position);
    }

    public float GetMinLidarDistance()
    {
        if (lidar == null) return 1e6f;
        return lidar.GetMinDistance();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxis("Vertical");
        ca[1] = Input.GetAxis("Horizontal");
    }
}