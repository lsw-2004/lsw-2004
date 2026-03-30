using UnityEngine;

/// <summary>
/// 环绕自身在水平面上发射射线，模拟 2D LiDAR。
/// 改动说明：
/// 1. 从 180° / 180 rays 改为 360° / 360 rays
/// 2. 其余扫描逻辑保持不变
/// </summary>
public class SimpleLidar2D : MonoBehaviour
{
    [Header("几何参数")]
    public int numRays = 360;
    public float fieldOfView = 360f;
    public float maxDistance = 10f;

    [Header("扫描频率")]
    public float scanFrequency = 10f;

    [Header("碰撞层")]
    public LayerMask obstacleLayers = ~0;

    [Header("调试显示")]
    public bool drawDebugRays = true;

    [Header("是否在 Update 中自动扫描")]
    public bool scanInUpdate = false;

    public float[] Distances => _distances;

    private float[] _distances;
    private float _timer;

    void Awake()
    {
        AllocateBufferIfNeeded();
        FillDefaultDistance();
    }

    void Update()
    {
        if (!scanInUpdate) return;

        _timer += Time.deltaTime;
        float interval = 1f / Mathf.Max(0.01f, scanFrequency);
        if (_timer >= interval)
        {
            DoScan();
            _timer = 0f;
        }
    }

    public void ScanOnce()
    {
        DoScan();
    }

    public float GetMinDistance()
    {
        if (_distances == null || _distances.Length == 0) return maxDistance;

        float minD = maxDistance;
        for (int i = 0; i < _distances.Length; i++)
        {
            if (_distances[i] < minD) minD = _distances[i];
        }
        return minD;
    }

    private void AllocateBufferIfNeeded()
    {
        if (_distances == null || _distances.Length != numRays)
        {
            _distances = new float[numRays];
        }
    }

    private void FillDefaultDistance()
    {
        if (_distances == null) return;
        for (int i = 0; i < _distances.Length; i++)
        {
            _distances[i] = maxDistance;
        }
    }

    private void DoScan()
    {
        AllocateBufferIfNeeded();

        float angleStep = (numRays > 1) ? fieldOfView / (numRays - 1) : 0f;
        float startAngle = -fieldOfView * 0.5f;

        for (int i = 0; i < numRays; i++)
        {
            float angle = startAngle + i * angleStep;

            Quaternion rot = Quaternion.Euler(0f, angle, 0f);
            Vector3 localDir = rot * Vector3.forward;
            Vector3 dir = transform.TransformDirection(localDir);

            Ray ray = new Ray(transform.position, dir);
            float dist = maxDistance;
            Color c = Color.green;

            if (Physics.Raycast(ray, out RaycastHit hit, maxDistance, obstacleLayers))
            {
                dist = hit.distance;
                c = Color.red;
            }

            _distances[i] = dist;

            if (drawDebugRays)
            {
                Debug.DrawRay(ray.origin, dir * dist, c, 0.1f);
            }
        }
    }
}