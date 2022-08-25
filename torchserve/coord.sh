
echo "running ../tests/test_coordinator_frames.py from root comment me if you dont want"
cp ../tests/test_coordinator_frames.py tst__.py
cp coordinators/triangles_coordinator.py ./coordinator.py
python tst__.py
rm tst__.py coordinator.py

# echo "running coordinators/triangles_coordinator.py comment me if you dont want"
# cp coordinators/triangles_coordinator.py ./coordinator.py
# python coordinator.py
# rm coordinator.py
